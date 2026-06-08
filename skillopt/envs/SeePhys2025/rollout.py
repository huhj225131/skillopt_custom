from __future__ import annotations

import base64
import json
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.SeePhys2025.evaluator import evaluate
from skillopt.model import chat_target_messages, is_target_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_target_exec


def _image_to_data_uri(path: str) -> str:
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_system(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return (
        "You are a careful physics vision QA assistant. Use the provided skill and the attached images to answer the question. "
        "Reasoning is allowed, and you should state the final answer clearly in plain text. "
        "Do not rely on special tags for the answer."
        f"\n\n{skill_section}"
    ).rstrip()


def _build_messages(
    item: dict,
    skill_content: str,
    image_detail: str,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> tuple[list[dict], str, str]:
    system = _build_system(skill_content)
    image_names = ", ".join(os.path.basename(path) for path in item.get("image_paths", [])) or "image"
    user_text = (
        f"""## Question\n{item['question']}\n\n. Solve the question with image information.
       
- Reasoning: brief internal reasoning as needed.
- Final: your final answer here"""
    )
    if diagnostic_mode and diagnostic_instruction.strip():
        user_text += f"\n\n## Training Readout\n{diagnostic_instruction.strip()}"
    content: list[dict] = [{"type": "text", "text": user_text}]
    for path in item.get("image_paths", []):
        image_url = {"url": _image_to_data_uri(path)}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        content.append({"type": "image_url", "image_url": image_url})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    return messages, system, user_text


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current SeePhys2025 physics vision question.",
        preamble=(
            "Use this skill when answering the current SeePhys2025 question. Inspect all attached images carefully and return the final answer inside <answer>...</answer>."
        ),
    )


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _run_codex_once(
    *,
    pred_dir: str,
    item: dict,
    skill_content: str,
    model: str,
    timeout: int,
    image_detail: str,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    _messages, _system, user_text = _build_messages(
        item,
        skill_content,
        image_detail,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
    )
    task_parts = [user_text]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the same images carefully and correct the answer if needed."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=task_text,
        images=item.get("image_paths", []),
    )
    prompt = (
        "Use the `skillopt-target` skill available in this workspace. Read `task.md`, inspect the attached images, and answer the question. "
        "Write the reasoning and final answer directly in plain text."
    )
    final_message, raw = run_target_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
        images=item.get("image_paths", []),
    )
    return final_message or raw, raw, skill_md, task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    exec_timeout: int = 120,
    image_detail: str = "auto",
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": item.get("subtask") or item.get("task_type") or "seephys",
        "task_description": item["question"],
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
        "image_paths": item.get("image_paths", []),
        "gold_answer": item.get("answers", []),
    }
    try:
        response = ""
        system_prompt = ""
        user_text = ""
        conversation: list[dict] = []
        if is_target_exec_backend():
            from skillopt.model import azure_openai as _llm

            conversation = [
                {
                    "role": "user",
                    "content": item["question"] + "\n\n" + ", ".join(os.path.basename(path) for path in item.get("image_paths", [])),
                }
            ]
            for turn in range(max_turns):
                response, _raw, system_prompt, user_text = _run_codex_once(
                    pred_dir=os.path.join(out_root, "predictions", item_id),
                    item=item,
                    skill_content=skill_content,
                    model=_llm.TARGET_DEPLOYMENT,
                    timeout=exec_timeout,
                    image_detail=image_detail,
                    diagnostic_mode=diagnostic_mode if turn == 0 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 0 else "",
                    previous_response=response if turn > 0 else "",
                )
                conversation.append({"type": "message", "turn": turn + 1, "content": response})
                if "<answer>" in response.lower():
                    break
        else:
            messages, system_prompt, user_text = _build_messages(
                item,
                skill_content,
                image_detail,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            )
            conversation = [
                {
                    "role": "user",
                    "content": user_text,
                }
            ]
            pred_dir = os.path.join(out_root, "predictions", item_id)
            os.makedirs(pred_dir, exist_ok=True)
            for turn in range(max_turns):
                if turn == 0:
                    # Log outgoing API request and capture raw response/meta for debugging
                    try:
                        req_path = os.path.join(pred_dir, "target_api_request.json")
                        with open(req_path, "w", encoding="utf-8") as _reqf:
                            json.dump(messages, _reqf, ensure_ascii=False, indent=2)
                    except Exception:
                        try:
                            # Fallback: write repr to txt for non-serializable payloads
                            with open(os.path.join(pred_dir, "target_api_request.txt"), "w", encoding="utf-8") as _reqf:
                                _reqf.write(repr(messages))
                        except Exception:
                            pass
                    resp_text, resp_meta = chat_target_messages(
                        messages=messages,
                        max_completion_tokens=10000,
                        retries=5,
                        stage="rollout",
                        timeout=exec_timeout,
                    )
                    # Immediate terminal logging of the target response for debugging
                    try:
                        print("[SeePhys TARGET RESPONSE]", resp_text, flush=True)
                        print("[SeePhys TARGET META]", json.dumps(resp_meta, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                    try:
                        resp_path = os.path.join(pred_dir, "target_api_response.json")
                        with open(resp_path, "w", encoding="utf-8") as _respf:
                            json.dump({"text": resp_text, "meta": resp_meta}, _respf, ensure_ascii=False, indent=2)
                    except Exception:
                        try:
                            with open(os.path.join(pred_dir, "target_api_response.txt"), "w", encoding="utf-8") as _respf:
                                _respf.write(repr({"text": resp_text, "meta": resp_meta}))
                        except Exception:
                            pass
                else:
                    refinement_messages = [
                        messages[0],
                        messages[1],
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "Review the same images carefully and answer again. Keep the final answer concise and in plain text."},
                    ]
                    try:
                        req_path = os.path.join(pred_dir, "target_api_request_refine.json")
                        with open(req_path, "w", encoding="utf-8") as _reqf:
                            json.dump(refinement_messages, _reqf, ensure_ascii=False, indent=2)
                    except Exception:
                        try:
                            with open(os.path.join(pred_dir, "target_api_request_refine.txt"), "w", encoding="utf-8") as _reqf:
                                _reqf.write(repr(refinement_messages))
                        except Exception:
                            pass
                    resp_text, resp_meta = chat_target_messages(
                        messages=refinement_messages,
                        max_completion_tokens=10000,
                        retries=5,
                        stage="rollout",
                        timeout=exec_timeout,
                    )
                    # Immediate terminal logging of the target response for debugging (refinement)
                    try:
                        print("[SeePhys TARGET RESPONSE - REFINE]", resp_text, flush=True)
                        print("[SeePhys TARGET META - REFINE]", json.dumps(resp_meta, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                    try:
                        resp_path = os.path.join(pred_dir, "target_api_response_refine.json")
                        with open(resp_path, "w", encoding="utf-8") as _respf:
                            json.dump({"text": resp_text, "meta": resp_meta}, _respf, ensure_ascii=False, indent=2)
                    except Exception:
                        try:
                            with open(os.path.join(pred_dir, "target_api_response_refine.txt"), "w", encoding="utf-8") as _respf:
                                _respf.write(repr({"text": resp_text, "meta": resp_meta}))
                        except Exception:
                            pass
                response = resp_text
                conversation.append({"type": "message", "turn": turn + 1, "content": _as_text(resp_text)})
                if turn == 0:
                    break

        result["response"] = _as_text(response)
        result["agent_ok"] = True
        result["n_turns"] = len(conversation) - 1

        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)
        raw_response_text = _as_text(response)
        with open(os.path.join(pred_dir, "target_raw_response.txt"), "w", encoding="utf-8") as f:
            f.write(raw_response_text)

        eval_result = evaluate(raw_response_text, item.get("answers", []), question=item["question"])
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["hard"] = int(eval_result["hard"])
        result["soft"] = float(eval_result["soft"])
        if result["soft"] <= 0.0:
            result["fail_reason"] = f"predicted '{eval_result['predicted_answer']}' but expected one of {item.get('answers', [])}"

        judge_input = {
            "question": item["question"],
            "gold_answers": item.get("answers", []),
            "model_response": raw_response_text,
        }
        judge_debug = {
            "input": judge_input,
            "raw_output": eval_result.get("judge_text", ""),
            "parsed_output": {
                "hard": int(eval_result["hard"]),
                "soft": float(eval_result["soft"]),
                "reason": eval_result.get("reason", ""),
                "predicted_answer": eval_result.get("predicted_answer", ""),
                "gold_answers": eval_result.get("gold_answers", item.get("answers", [])),
            },
        }
        with open(os.path.join(pred_dir, "judge_debug.json"), "w", encoding="utf-8") as f:
            json.dump(judge_debug, f, ensure_ascii=False, indent=2)
        with open(os.path.join(pred_dir, "judge_debug.txt"), "w", encoding="utf-8") as f:
            f.write("[JUDGE INPUT]\n")
            f.write(json.dumps(judge_input, ensure_ascii=False, indent=2))
            f.write("\n\n[JUDGE RAW OUTPUT]\n")
            f.write(str(eval_result.get("judge_text", "")))
            f.write("\n\n[JUDGE PARSED OUTPUT]\n")
            f.write(json.dumps(judge_debug["parsed_output"], ensure_ascii=False, indent=2))

        eval_detail = (
            "[EVALUATION RESULT]\n"
            f"Question: {item['question']}\n"
            f"Predicted answer: {eval_result['predicted_answer']!r}\n"
            f"Gold answers: {item.get('answers', [])!r}\n"
            f"Hard: {int(eval_result['hard'])}\n"
            f"Soft: {float(eval_result['soft']):.4f}\n"
            f"Judge: {eval_result.get('reason', '')}"
        )
        conversation.append({"role": "system", "content": eval_detail})
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"
        result["traceback"] = traceback.format_exc(limit=5)
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    exec_timeout: int = 120,
    workers: int = 16,
    image_detail: str = "auto",
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> list[dict]:
    if not items:
        return []
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_turns=max_turns,
                exec_timeout=exec_timeout,
                image_detail=image_detail,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            ): item
            for item in items
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
    order = {str(item["id"]): idx for idx, item in enumerate(items)}
    results.sort(key=lambda row: order.get(str(row.get("id")), 10**9))
    return results