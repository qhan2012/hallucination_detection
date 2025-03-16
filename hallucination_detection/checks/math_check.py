# hallucination_detection/checks/math_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO

class MathCheck(BaseCheck):
    """
    Check for mathematical statements using LLM verification.
    """
    def __init__(self):
        super().__init__()
        self.llm_container.register_llm("cerebras", "llama3.3-70b")

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[MathCheck] Checking math problem: {text}")
        
        prompt_template = """Analyze the following mathematical statement and determine if it is correct.
        Rate it from 0 (completely incorrect) to 1 (completely correct).
        Only respond with a number between 0 and 1, nothing else.
        Pay attention to numerical calculations, inequalities, and mathematical properties.
        Be conservative - if you're not completely sure, give a lower score.
        
        Statement: {text}
        Correctness score:"""
        
        score = self.get_llm_truth_score(text, prompt_template, "cerebras", "llama3.3-70b")
        debug_print(DEBUG_INFO, f"[MathCheck] Score for '{text}': {score}")
        return score
