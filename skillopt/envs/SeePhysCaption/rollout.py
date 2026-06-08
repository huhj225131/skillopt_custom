from __future__ import annotations

import base64
import json
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.SeePhysCaption.evaluator import evaluate
from skillopt.model import chat_target_messages


def _image_to_data_uri(path: str) -> str:
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_system(skill_content: str) -> str:
    skill_section = f"## Skill Instructions\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return (
        "You are an expert physics image captioner. Your task is to look at the provided image(s) and a brief problem text, "
        "and generate a highly detailed description that restores all missing physical quantities, labels, and parameters found in the image. "
        "State your final description plainly."
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
    
    user_text = (
        f"## Input Problem\n{item['input_text']}\n\n"
        "Analyze the provided images and supplement the missing physical/geometric information into the text above to create a complete and detailed caption."
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


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


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
        "task_type": item.get("subtask") or item.get("task_type") or "seephys_caption",
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
        "reference_text": item.get("ground_truth", item.get("answers", [""])[0] if item.get("answers") else ""),
    }
    try:
        messages, system_prompt, user_text = _build_messages(
            item,
            skill_content,
            image_detail,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
        )
        
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save API request
        try:
            req_path = os.path.join(pred_dir, "target_api_request.json")
            with open(req_path, "w", encoding="utf-8") as _reqf:
                json.dump(messages, _reqf, ensure_ascii=False, indent=2)
        except Exception:
            pass
            
        # Call Qwen
        resp_text, resp_meta = chat_target_messages(
            messages=messages,
            max_completion_tokens=32000,
            retries=5,
            stage="rollout",
            timeout=None,
            enable_thinking=True,
        )
        
        # Save logs
        with open(os.path.join(pred_dir, "target_api_response.json"), "w", encoding="utf-8") as f:
            json.dump({"text": resp_text, "meta": resp_meta}, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
            
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)
            
        raw_response_text = _as_text(resp_text)
        with open(os.path.join(pred_dir, "target_raw_response.txt"), "w", encoding="utf-8") as f:
            f.write(raw_response_text)

        result["response"] = raw_response_text
        result["predicted_answer"] = raw_response_text
        result["agent_ok"] = True
        result["n_turns"] = 1

        # Hybrid Evaluator Check
        eval_result = evaluate(raw_response_text, item.get("answers", []), question=item["question"])
        
        result["hard"] = int(eval_result["hard"])
        result["soft"] = float(eval_result["soft"])
        result["fail_reason"] = eval_result.get("reason", "")
        
        judge_debug = {
            "input": {
                "question": item["question"],
                "gold_answers": item.get("answers", []),
                "model_response": raw_response_text,
            },
            "parsed_output": {
                "hard": int(eval_result["hard"]),
                "soft": float(eval_result["soft"]),
                "reason": eval_result.get("reason", ""),
                "metrics": eval_result.get("metrics", {}),
            },
        }
        with open(os.path.join(pred_dir, "judge_debug.json"), "w", encoding="utf-8") as f:
            json.dump(judge_debug, f, ensure_ascii=False, indent=2)

        # Build conversation trace for the Reflect stage
        eval_detail = (
            "[EVALUATION RESULT]\n"
            f"Question: {item['question']}\n"
            f"Predicted answer: {result['predicted_answer']!r}\n"
            f"Gold answers: {item.get('answers', [])!r}\n"
            f"Hard: {result['hard']}\n"
            f"Soft: {result['soft']:.4f}\n"
            f"Judge: {result['fail_reason']}"
        )
        conversation = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": raw_response_text},
            {"role": "system", "content": eval_detail}
        ]
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

    except Exception as e:
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
