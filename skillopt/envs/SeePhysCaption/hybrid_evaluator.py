import re
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure nltk packages are downloaded at runtime if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class HybridCaptionEvaluator:
    def __init__(self, rouge_weight=0.3, bleu_weight=0.3, exact_num_weight=0.4, pass_threshold=0.75):
        """
        Initialize the evaluator with weights for different metrics.
        exact_num_weight is given highest weight due to the physics domain.
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        
        self.rouge_weight = rouge_weight
        self.bleu_weight = bleu_weight
        self.exact_num_weight = exact_num_weight
        self.pass_threshold = pass_threshold
        
    def _extract_physics_numbers(self, text):
        """
        Extract numerical values (integers and floats) from the given text.
        """
        if not text:
            return set()
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
        return set(matches)
        
    def evaluate(self, generated_caption: str, ground_truth: str) -> dict:
        """
        Evaluates the generated caption against the ground truth and returns SkillOpt format.
        """
        generated_caption = str(generated_caption or "")
        ground_truth = str(ground_truth or "")
        
        if not ground_truth:
            return {
                "hard": 1 if generated_caption else 0,
                "soft": 1.0 if generated_caption else 0.0,
                "reason": "Missing ground truth to compare.",
            }

        # 1. Calculate ROUGE-L
        rouge_scores = self.rouge_scorer.score(ground_truth, generated_caption)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        
        # 2. Calculate BLEU Score
        ref_tokens = nltk.word_tokenize(ground_truth.lower())
        gen_tokens = nltk.word_tokenize(generated_caption.lower())
        
        # Guard against empty tokens
        if not gen_tokens:
            bleu_score = 0.0
        else:
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothie)
        
        # 3. Calculate Number Match Score
        gt_nums = self._extract_physics_numbers(ground_truth)
        gen_nums = self._extract_physics_numbers(generated_caption)
        
        if len(gt_nums) == 0:
            num_match_score = 1.0
            missing_nums = set()
        else:
            matched_nums = gt_nums.intersection(gen_nums)
            missing_nums = gt_nums - gen_nums
            num_match_score = len(matched_nums) / len(gt_nums)
            
        # 4. Soft Score
        soft_score = (
            self.rouge_weight * rouge_l_f1 + 
            self.bleu_weight * bleu_score + 
            self.exact_num_weight * num_match_score
        )
        
        # 5. Hard Score (Rule-based pre-check)
        if soft_score >= self.pass_threshold and len(missing_nums) == 0:
            pre_hard_score = 1
        else:
            pre_hard_score = 0
            
        # 6. LLM Judge Augmentation
        from skillopt.model import chat_target_messages
        from skillopt.utils import extract_json
        
        system_prompt = (
            "You are a strict physics caption evaluator. Compare the Generated Caption with the Ground Truth. "
            "Evaluate physical correctness, semantic equivalence, and completeness. "
            "Return only JSON: {\"hard\": 0 or 1, \"soft\": float between 0 and 1, \"reason\": \"your detailed explanation\"}."
            "Do NOT include any markdown formatting, only the JSON object."
        )
        user_prompt = f"Ground Truth:\n{ground_truth}\n\nGenerated Caption:\n{generated_caption}\n\n"
        user_prompt += f"Automatic Metrics:\n- ROUGE-L: {rouge_l_f1:.2f}\n- BLEU: {bleu_score:.2f}\n"
        if missing_nums:
            user_prompt += f"- Potentially Missing Numbers: {list(missing_nums)}\n\nIf the generated caption is genuinely missing these numbers (and not just phrasing them differently), you MUST set hard=0.\n"
        else:
            user_prompt += "- Missing Numbers: None\n\n"
            
        user_prompt += "Explain what the generated caption missed or got wrong, and then give the final hard and soft scores."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            resp_text, _ = chat_target_messages(
                messages=messages,
                max_completion_tokens=32000,
                retries=3,
                stage="seephys_judge",
                enable_thinking=True
            )
            parsed = extract_json(resp_text)
            if isinstance(parsed, dict):
                hard_score = int(parsed.get("hard", pre_hard_score))
                llm_soft = float(parsed.get("soft", soft_score))
                reason = str(parsed.get("reason", ""))
                # Blend soft score: average of rules and LLM
                soft_score = (soft_score + llm_soft) / 2.0
            else:
                raise ValueError("LLM returned invalid JSON")
        except Exception as e:
            # Fallback to rule-based if LLM fails
            hard_score = pre_hard_score
            reason_parts = []
            if len(missing_nums) > 0:
                reason_parts.append(f"Model missed or incorrectly identified important physical parameters: {list(missing_nums)}.")
            if rouge_l_f1 < 0.6:
                reason_parts.append(f"Sentence structure/vocabulary is inaccurate (ROUGE-L: {rouge_l_f1:.2f}).")
            if bleu_score < 0.4:
                reason_parts.append(f"Unnatural expression or incorrect phrasing (BLEU: {bleu_score:.2f}).")
            if hard_score == 1:
                reason = "Success. Physical parameters and sentence structure match well. (Fallback Judge)"
            else:
                reason = "Failure. " + " ".join(reason_parts) + f" (Fallback Judge: {e})"
            
        return {
            "hard": hard_score,
            "soft": round(soft_score, 4),
            "reason": reason,
            "metrics": {
                "rouge_l": round(rouge_l_f1, 4),
                "bleu": round(bleu_score, 4),
                "num_match": round(num_match_score, 4)
            }
        }
