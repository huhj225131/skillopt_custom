from __future__ import annotations

from typing import Any
from skillopt.envs.SeePhysCaption.hybrid_evaluator import HybridCaptionEvaluator

# Initialize globally to avoid recreating the smoothing function repeatedly
_hybrid_evaluator = HybridCaptionEvaluator()

def evaluate(prediction_text: str, gold: Any, question: str = "") -> dict[str, Any]:
    """
    Evaluates the generated caption against the gold truth using the hybrid metrics.
    """
    # SkillOpt often passes single items or lists. Get the string.
    gold_text = ""
    if isinstance(gold, list) and gold:
        gold_text = str(gold[0])
    else:
        gold_text = str(gold)
        
    prediction_text = str(prediction_text or "")
    
    # Run the hybrid evaluator
    result = _hybrid_evaluator.evaluate(prediction_text, gold_text)
    
    # Format output for SkillOpt
    return {
        "hard": result["hard"],
        "soft": result["soft"],
        "reason": result["reason"],
        "predicted_answer": prediction_text,
        "gold_answers": [gold_text],
        "judge_text": str(result.get("metrics", {})),
    }

class SeePhysCaptionEvaluator:
    def evaluate(self, prediction_text: str, gold: Any, question: str = "") -> dict[str, Any]:
        return evaluate(prediction_text, gold, question)
