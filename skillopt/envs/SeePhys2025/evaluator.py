from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Iterable
from typing import Any

from skillopt.model.qwen_backend import chat_target_messages as qwen_chat_target_messages
from skillopt.model.qwen_backend import TARGET_DEPLOYMENT as QWEN_TARGET_DEPLOYMENT
from skillopt.utils import extract_json

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_JUDGE_SYSTEM = (
    "You are a strict answer judge for SeePhys2025 physics vision QA. "
    "Compare the gold answer(s) with the model response and decide whether the model's final answer is acceptable. "
    "Treat mathematically equivalent answers as correct, even if formatting differs. "
    "Use the gold answer as the reference truth; do not invent a new answer. "
    "Return only a JSON object, with no markdown fences and no extra text. "
    "Use exactly this schema: {\"hard\": 0|1, \"soft\": 0|1, \"reason\": string}. "
    "Set hard and soft to the same value: 1 for acceptable/correct, 0 for unacceptable/wrong. "
    "Use reason to explain the decision briefly. "
    "Example expected response (return this JSON only): {\"hard\": 1, \"soft\": 1, \"reason\": \"Numeric match and reasoning supports the result\"}. "
    "Do NOT include any other text, explanations, or markdown fences — only the JSON object."
)
_DEBUG_JUDGE_IO = os.environ.get("SEEPHYS_DEBUG_JUDGE_IO", "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("$", "")
    text = text.replace(",", "")
    return text


def _extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _coerce_targets(raw: Any) -> list[str]:
    if raw is None:
        return [""]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [""]
        if text[:1] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
            if parsed is not None:
                return _coerce_targets(parsed)
        return [text]
    if isinstance(raw, dict):
        for key in ("answer", "answers", "ground_truth"):
            if key in raw:
                return _coerce_targets(raw[key])
        return [str(raw)]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
        values: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                for key in ("answer", "text", "value"):
                    if key in item:
                        values.extend(_coerce_targets(item[key]))
                        break
                else:
                    values.append(str(item))
            else:
                values.append(str(item))
        return values or [""]
    return [str(raw)]


def _build_messages(question: str, prediction_text: str, gold_answers: list[str]) -> list[dict[str, Any]]:
    payload = {
        "question": question,
        "gold_answers": gold_answers,
        "model_response": prediction_text,
    }
    user_text = (
        "Judge the model response against the gold answer(s).\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return only JSON in this exact shape: {\"hard\": 0|1, \"soft\": 0|1, \"reason\": string}. "
        "Do not wrap the JSON in markdown fences and do not add any surrounding prose. "
        "Judge both the reasoning and the final answer, and only mark correct when the reasoning supports the answer."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def _debug_judge_io(messages: list[dict[str, Any]], raw_text: str | None = None) -> None:
    if not _DEBUG_JUDGE_IO:
        return
    print("[SeePhys2025 judge] input messages:", flush=True)
    print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
    if raw_text is not None:
        print("[SeePhys2025 judge] raw output:", flush=True)
        print(raw_text, flush=True)


def _run_judge(messages: list[dict[str, Any]]) -> str:
    raw, _meta = qwen_chat_target_messages(
        messages=messages,
        max_completion_tokens=10000,
        retries=3,
        stage="seephys_judge",
        timeout=60,
        enable_thinking=False,
    )
    return str(raw)


def _parse_judge_output(raw_text: str) -> dict[str, Any] | None:
    parsed = extract_json(raw_text)
    if not isinstance(parsed, dict):
        return None
    try:
        hard = 1 if int(parsed.get("hard", 0)) else 0
    except Exception:
        hard = 0
    try:
        soft = float(parsed.get("soft", hard))
    except Exception:
        soft = float(hard)
    soft = 1.0 if soft >= 1.0 else 0.0
    reason = str(parsed.get("reason", "")).strip()
    if soft != float(hard):
        soft = float(hard)
    return {"hard": hard, "soft": soft, "reason": reason, "raw": raw_text}


def _fallback_exact_match(prediction_text: str, gold: Any) -> dict[str, Any]:
    targets = _coerce_targets(gold)
    predicted_norm = _normalize_text(prediction_text)
    target_norms = [_normalize_text(target) for target in targets]
    exact = float(bool(predicted_norm) and predicted_norm in target_norms)
    if not predicted_norm and any(not target for target in target_norms):
        exact = 1.0
    return {
        "hard": int(exact >= 1.0),
        "soft": int(exact >= 1.0),
        "reason": "fallback exact-match scoring",
        "predicted_answer": prediction_text,
        "gold_answers": targets,
    }


def extract_final_answer(text: str) -> str:
    # Look for patterns like "- Final: <answer>", "Final: <answer>", "**Final:** <answer>", "Final Answer: <answer>"
    pattern = r"(?:-\s*)?\(?\*?Final(?:\s*Answer)?\*?\)?[：:\s]\s*(.*)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        ans = matches[-1].strip()
        ans = ans.strip("*\"' ")
        return ans
    # Fallback to the last line if not found
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def evaluate(prediction_text: str, gold: Any, question: str = "") -> dict[str, Any]:
    targets = _coerce_targets(gold)
    extracted = extract_final_answer(prediction_text)
    
    predicted_norm = _normalize_text(extracted)
    target_norms = [_normalize_text(target) for target in targets]
    
    # Check if the prediction matches any of the target norms
    is_correct = bool(predicted_norm) and predicted_norm in target_norms
    
    # Handle the case where both are empty/null
    if not predicted_norm and any(not target for target in target_norms):
        is_correct = True
        
    hard = 1 if is_correct else 0
    soft = 1.0 if is_correct else 0.0
    
    # English reason formatting
    if is_correct:
        reason = f"✅ Correct prediction! (Predicted: '{extracted}' == Ground Truth: '{gold}')"
    else:
        reason = f"❌ Incorrect prediction! (Predicted: '{extracted}' | Ground Truth: '{gold}')"
    
    return {
        "hard": hard,
        "soft": soft,
        "reason": reason,
        "predicted_answer": extracted,
        "gold_answers": targets,
        "judge_text": "regex scoring",
    }