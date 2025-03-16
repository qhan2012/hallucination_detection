# hallucination_detection/checks/base_check.py

from abc import ABC, abstractmethod
from ..debug_logger import debug_print, DEBUG_INFO
from ..llm import LLMContainer

class BaseCheck(ABC):
    """
    Abstract base class for all checks.
    """

    def __init__(self):
        self.llm_container = LLMContainer()

    @abstractmethod
    def check_fact(self, text: str) -> float:
        """
        Implementations should return a "believability" or "confidence" score 
        indicating how likely the statement is true or not.
        """
        pass

    def get_llm_truth_score(self, text: str, prompt_template: str, llm: str = "cerebras", model: str = "llama3.1-8b") -> float:
        """
        Get truth score from LLM for a given text using specified prompt template.
        
        Args:
            text: Text to analyze
            prompt_template: Prompt template with {text} placeholder
            model: LLM model to use
            
        Returns:
            float: Truth score between 0 and 1
        """
        llm_client = self.llm_container.get_llm(llm, model)
        prompt = prompt_template.format(text=text)
        
        response = llm_client.generate_text(prompt)
        
        try:
            debug_print(DEBUG_INFO, f"LLM returned response: {response}")
            score = float(response.strip())
            score = max(0.0, min(1.0, score))
            debug_print(DEBUG_INFO, f"LLM returned truth score: {score}")
        except ValueError:
            debug_print(DEBUG_INFO, f"Failed to parse LLM response, using default score")
            score = 0.5
            
        return score
