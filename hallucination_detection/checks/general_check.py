# hallucination_detection/checks/general_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO

class GeneralCheck(BaseCheck):
    """
    Check for general facts using LLM verification.
    """
    def __init__(self):
        super().__init__()
        self.llm_container.register_llm("cerebras", "llama3.1-8b")

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[GeneralCheck] Checking general fact: {text}")
        
        prompt_template = """Analyze the following general statement and determine its truthfulness.
        Rate it from 0 (completely false) to 1 (completely true).
        Only respond with a number between 0 and 1, nothing else.
        Consider common knowledge, real-world facts, and general information.
        If you're not completely sure, give a moderate score around 0.5.
        
        Statement: {text}
        Truth score:"""
        
        score = self.get_llm_truth_score(text, prompt_template)
        debug_print(DEBUG_INFO, f"[GeneralCheck] Score for '{text}': {score}")
        return score


class NoneCheck(BaseCheck):
    """
    Example check for general facts, e.g. using Wikipedia or knowledge graphs.
    """

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[NoneCheck] This is point of view or adjective sentence，: {text}")
        # Integrate with Wikipedia or knowledge graph
        return 1 
