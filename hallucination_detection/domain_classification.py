# hallucination_detection/domain_classification.py

from typing import Optional
from .debug_logger import debug_print, DEBUG_INFO
from .llm import LLMContainer

class DomainClassifier:
    """
    Classifies the domain of a statement using LLM into categories such as
    'history', 'paper', 'math', 'logic', 'general', 'latest_news', etc.
    """

    def __init__(self):
        self.llm_container = LLMContainer()
        self.llm_container.register_llm("cerebras", "llama3.3-70b")
        self.domains = ["history", "paper", "math", "logic", "latest_news", "general", "none"]

    def classify(self, text: str) -> Optional[str]:
        """
        Uses LLM to classify the domain of the input text.
        """
        debug_print(DEBUG_INFO, f"Classifying domain for text: {text}")

        llm_client = self.llm_container.get_llm("cerebras", "llama3.3-70b")
        
        prompt = f"""Classify the following text into one of these domains: {', '.join(self.domains)}
        Only respond with the domain name, nothing else. If it is just a point of view or adjective sentence， return 'none'. Be conservative with math category except there is clear math formula. Pay attention on the reference which can be in paper category.
        
        Text: {text}"""

        response = llm_client.generate_text(prompt)

        # Extract content from ChatCompletionResponse
        content = response.choices[0].message.content if hasattr(response, 'choices') else response

        # Clean up response to get just the domain
        debug_print(DEBUG_INFO, f"LLM return classification: {content}")
        predicted_domain = content.strip().lower()
        
        # Validate the response is one of our expected domains
        if predicted_domain in self.domains:
            return predicted_domain
        
        # Fallback to general if LLM response is not in expected domains
        return "general"
