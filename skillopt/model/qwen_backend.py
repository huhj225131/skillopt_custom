"""OpenAI-compatible Qwen chat backend for the target path."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
import tempfile
import threading as _thr

import httpx
from openai import OpenAI

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    TokenTracker,
    default_model_for_backend,
)

BASE_URL = os.environ.get("QWEN_CHAT_BASE_URL", "http://localhost:8001/v1")
API_KEY = os.environ.get("QWEN_CHAT_API_KEY", "")
TIMEOUT_SECONDS = float(os.environ.get("QWEN_CHAT_TIMEOUT_SECONDS", "300") or 300)
MAX_TOKENS = int(os.environ.get("QWEN_CHAT_MAX_TOKENS", "10000") or 10000)
TEMPERATURE: float | None = None
_raw_temperature = os.environ.get("QWEN_CHAT_TEMPERATURE", "0.7").strip()
if _raw_temperature:
    TEMPERATURE = float(_raw_temperature)
ENABLE_THINKING = os.environ.get("QWEN_CHAT_ENABLE_THINKING", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

TARGET_DEPLOYMENT = os.environ.get(
    "TARGET_DEPLOYMENT",
    default_model_for_backend("qwen_chat"),
)
OPTIMIZER_DEPLOYMENT = os.environ.get(
    "OPTIMIZER_DEPLOYMENT",
    default_model_for_backend("qwen_chat"),
)

_config_lock = threading.Lock()
tracker = TokenTracker()
_client: OpenAI | None = None
DEBUG_DIR = ""


def _chat_url() -> str:
    base = BASE_URL.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


_clients: list[OpenAI] = []
_client_index = 0
_last_initialized_base_url = ""


def _get_client() -> OpenAI:
    global _clients, _client_index, _last_initialized_base_url
    with _config_lock:
        urls = [u.strip() for u in BASE_URL.split(",") if u.strip()]
        if not urls:
            urls = ["http://localhost:8000/v1"]
            
        if not _clients or BASE_URL != _last_initialized_base_url:
            _clients = []
            for url in urls:
                c = OpenAI(
                    api_key=API_KEY or "dummy",
                    base_url=url,
                    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
                    max_retries=5,
                )
                _clients.append(c)
            _client_index = 0
            _last_initialized_base_url = BASE_URL
            
        client = _clients[_client_index]
        _client_index = (_client_index + 1) % len(_clients)
        return client


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(value)


def _usage_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compat_message_from_payload(message: dict[str, Any], choice: dict[str, Any]) -> CompatAssistantMessage:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    tool_calls: list[CompatToolCall] = []
    for index, tool_call in enumerate(message.get("tool_calls") or [], start=1):
        function = tool_call.get("function") or {}
        tool_calls.append(
            CompatToolCall(
                id=str(tool_call.get("id") or f"qwen_tool_{index}"),
                type=str(tool_call.get("type") or "function"),
                function=CompatToolFunction(
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                ),
            )
        )
    return CompatAssistantMessage(
        content=content,
        tool_calls=tool_calls,
        metadata={
            "finish_reason": choice.get("finish_reason"),
            "choice0": _json_safe(choice),
        },
    )


def _extract_chunk_text(chunk: Any) -> str:
    try:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        choice0 = choices[0]
        delta = getattr(choice0, "delta", None)
        if delta is None and isinstance(choice0, dict):
            delta = choice0.get("delta")
        reasoning_parts: list[str] = []
        content = getattr(delta, "content", None) if delta is not None else None
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        reasoning_content = getattr(delta, "reasoning_content", None) if delta is not None else None
        if reasoning_content is None and isinstance(delta, dict):
            reasoning_content = delta.get("reasoning_content")
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or json.dumps(part, ensure_ascii=False)))
                else:
                    parts.append(str(part))
            content_text = "".join(parts)
        elif isinstance(delta, str):
            content_text = delta
        else:
            content_text = ""

        if isinstance(reasoning_content, str) and reasoning_content.strip():
            reasoning_parts.append(reasoning_content)
        elif isinstance(reasoning_content, list):
            for part in reasoning_content:
                if isinstance(part, str):
                    reasoning_parts.append(part)
                elif isinstance(part, dict):
                    reasoning_parts.append(str(part.get("text") or part.get("content") or json.dumps(part, ensure_ascii=False)))
                else:
                    reasoning_parts.append(str(part))

        reasoning_text = "".join(reasoning_parts).strip()
        if content_text.strip():
            return content_text
        if reasoning_text:
            return reasoning_text
        return ""
    except Exception:
        return ""


def _post_chat_completion(payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    client = _get_client()
    create_kwargs: dict[str, Any] = {
        "model": payload["model"],
        "messages": payload["messages"],
        "max_tokens": payload["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
    }
    if timeout is not None:
        create_kwargs["timeout"] = timeout
    if "temperature" in payload:
        create_kwargs["temperature"] = payload["temperature"]
    if "tools" in payload:
        create_kwargs["tools"] = payload["tools"]
    if "tool_choice" in payload:
        create_kwargs["tool_choice"] = payload["tool_choice"]

    stream = client.chat.completions.create(**create_kwargs)
    chunks: list[Any] = []
    text_parts: list[str] = []
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_finish_reason: str | None = None
    for chunk in stream:
        chunks.append(_json_safe(chunk))
        try:
            if getattr(chunk, "usage", None):
                usage_info = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
                }
            choices = getattr(chunk, "choices", None) or []
            if choices:
                choice0 = choices[0]
                last_finish_reason = getattr(choice0, "finish_reason", None) or last_finish_reason
                part = _extract_chunk_text(chunk)
                if part:
                    text_parts.append(part)
        except Exception:
            pass

    text = "".join(text_parts)
    if not usage_info["total_tokens"]:
        usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
    return {
        "text": text,
        "usage": usage_info,
        "chunks": chunks,
        "finish_reason": last_finish_reason,
        "payload": payload,
    }


def _chat_messages_impl(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    deployment: str | None = None,
    timeout: float | None = None,
    enable_thinking: bool | None = None,
) -> tuple[Any, dict[str, int]]:
    payload: dict[str, Any] = {
        "model": deployment or TARGET_DEPLOYMENT,
        "messages": _json_safe(messages),
        "max_tokens": min(max_completion_tokens, MAX_TOKENS),
    }
    thinking_opt = enable_thinking if enable_thinking is not None else ENABLE_THINKING
    payload["chat_template_kwargs"] = {"enable_thinking": thinking_opt}
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    if tools:
        payload["tools"] = _json_safe(tools)
        if tool_choice is not None:
            payload["tool_choice"] = _json_safe(tool_choice)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            data = _post_chat_completion(payload, timeout)
            # Optionally print full payload/response to terminal for immediate inspection
            if os.environ.get("QWEN_DEBUG_PRINT", "").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    print("[QWEN DEBUG PAYLOAD]", flush=True)
                    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
                    print("[QWEN DEBUG RESPONSE]", flush=True)
                    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
                except Exception:
                    pass
            # Persistent debug directory if configured
            if DEBUG_DIR:
                try:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    is_optimizer = stage in {"optimizer", "analyst", "aggregate", "ranking", "slow_update", "meta_skill"}
                    prefix = "optimizer" if is_optimizer else "target"
                    
                    # Log request
                    req_name = f"{prefix}_api_request_{stage}_{int(time.time() * 1000)}_{_thr.get_ident()}.json"
                    with open(os.path.join(DEBUG_DIR, req_name), "w", encoding="utf-8") as df:
                        json.dump(payload, df, ensure_ascii=False, indent=2)
                        
                    # Log response
                    resp_name = f"{prefix}_api_response_{stage}_{int(time.time() * 1000)}_{_thr.get_ident()}.json"
                    with open(os.path.join(DEBUG_DIR, resp_name), "w", encoding="utf-8") as df:
                        json.dump(data, df, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            # Optional debug: persist full request/response to temp when requested
            if os.environ.get("QWEN_DEBUG_IO", "").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    debug_path = tempfile.gettempdir()
                    fname = f"qwen_debug_{stage}_{int(time.time())}_{_thr.get_ident()}.json"
                    with open(os.path.join(debug_path, fname), "w", encoding="utf-8") as df:
                        json.dump({"payload": payload, "response": data}, df, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            text = str(data.get("text") or "")
            usage_info = data.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            # If extracted text is empty, persist the full request/response for offline inspection
            if not text or str(text).strip() == "":
                try:
                    debug_path = tempfile.gettempdir()
                    fname = f"qwen_debug_empty_response_{stage}_{int(time.time())}_{_thr.get_ident()}.json"
                    dbg = {"payload": payload, "response": data}
                    with open(os.path.join(debug_path, fname), "w", encoding="utf-8") as df:
                        json.dump(dbg, df, ensure_ascii=False, indent=2)
                    print(f"[QWEN DEBUG] empty extracted text — dumped full response to {os.path.join(debug_path, fname)}", flush=True)
                except Exception:
                    pass
            if return_message:
                return CompatAssistantMessage(
                    content=text,
                    tool_calls=[],
                    metadata={
                        "finish_reason": data.get("finish_reason"),
                        "chunks": data.get("chunks", []),
                    },
                ), usage_info
            return text, usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Qwen chat call failed after {retries} retries: {last_err}")


def configure_qwen_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
    debug_dir: str | None = None,
) -> None:
    global BASE_URL, API_KEY, TEMPERATURE, TIMEOUT_SECONDS, MAX_TOKENS, ENABLE_THINKING, DEBUG_DIR, _client
    with _config_lock:
        if debug_dir is not None:
            DEBUG_DIR = str(debug_dir).strip()
        if base_url is not None:
            BASE_URL = str(base_url).strip() or BASE_URL
            os.environ["QWEN_CHAT_BASE_URL"] = BASE_URL
        if api_key is not None:
            API_KEY = str(api_key).strip()
            os.environ["QWEN_CHAT_API_KEY"] = API_KEY
        if temperature is not None:
            raw = str(temperature).strip()
            TEMPERATURE = float(raw) if raw else None
            os.environ["QWEN_CHAT_TEMPERATURE"] = raw
        if timeout_seconds is not None:
            TIMEOUT_SECONDS = float(timeout_seconds)
            os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = str(timeout_seconds)
        if max_tokens is not None:
            MAX_TOKENS = int(max_tokens)
            os.environ["QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
        if enable_thinking is not None:
            if isinstance(enable_thinking, str):
                ENABLE_THINKING = enable_thinking.strip().lower() in {"1", "true", "yes", "on"}
            else:
                ENABLE_THINKING = bool(enable_thinking)
            os.environ["QWEN_CHAT_ENABLE_THINKING"] = "true" if ENABLE_THINKING else "false"
        _client = None


def get_max_tokens() -> int:
    return MAX_TOKENS


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    enable_thinking: bool | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    enable_thinking: bool | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        deployment=OPTIMIZER_DEPLOYMENT,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
    enable_thinking: bool | None = None,
) -> tuple[Any, dict[str, int]]:
    del reasoning_effort
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
    enable_thinking: bool | None = None,
) -> tuple[Any, dict[str, int]]:
    del reasoning_effort
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        deployment=OPTIMIZER_DEPLOYMENT,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    del effort


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment or default_model_for_backend("qwen_chat")
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_DEPLOYMENT


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment or default_model_for_backend("qwen_chat")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_DEPLOYMENT
