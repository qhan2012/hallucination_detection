# hallucination_detection/checks/logic_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO

class LogicCheck(BaseCheck):
    """Check for logical statements using LLM verification."""
    
    def __init__(self):
        super().__init__()
        self.llm_container.register_llm("cerebras", "llama3.1-8b")
        self.llm_container.register_llm("cerebras", "llama3.3-70b")

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[LogicCheck] Checking logical statement: {text}")
        
        prompt_template = """Analyze the following logical statement and determine its validity.
        Rate it from 0 (completely invalid) to 1 (completely valid).
        Only respond with a number between 0 and 1, nothing else.
        
        Statement: {text}
        Validity score:"""
        
        score = self.get_llm_truth_score(text, prompt_template, "cerebras", "llama3.3-70b")
        debug_print(DEBUG_INFO, f"[LogicCheck] Score for '{text}': {score}")
        return score
