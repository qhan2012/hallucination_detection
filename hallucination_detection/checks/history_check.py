# hallucination_detection/checks/history_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO
from ..llm import LLMContainer

class HistoryCheck(BaseCheck):
    """
    Check for historical statements using LLM verification.
    """
    def __init__(self):
        super().__init__()
        self.llm_container = LLMContainer()
        self.llm_container.register_llm("cerebras", "llama3.1-8b")

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[HistoryCheck] Checking historical fact: {text}")
        
        prompt_template = """Analyze the following historical statement and determine its truthfulness.
        Rate it from 0 (completely false) to 1 (completely true).
        Only respond with a number between 0 and 1, nothing else.
        
        Statement: {text}
        Truth score:"""
        
        score = self.get_llm_truth_score(text, prompt_template)
        debug_print(DEBUG_INFO, f"[HistoryCheck] Score for '{text}': {score}")
        return score
