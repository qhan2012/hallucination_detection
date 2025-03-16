# hallucination_detection/llm.py
"""
This module provides a container (LLMContainer) that holds multiple LLM clients.
A client can specify which LLM engine and model to use.
"""

from typing import Dict
from .debug_logger import debug_print, DEBUG_INFO, DEBUG_VERBOSE
import os
from cerebras.cloud.sdk import Cerebras

class LLMClient:
    """
    Mock implementation of a generic LLM client.
    In practice, you might connect to OpenAI, Anthropic, or any other service.
    """
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        debug_print(DEBUG_VERBOSE, f"Initialized LLMClient with name={name}, model={model}")

    def generate_text(self, prompt: str) -> str:
        debug_print(DEBUG_INFO, f"LLMClient generating text for prompt: {prompt}")
        
        if self.name.lower() == "openai":
            response = f"[OpenAI-{self.model}] Processing with GPT: {prompt}"
        elif self.name.lower() == "anthropic":
            response = f"[Anthropic-{self.model}] Processing with Claude: {prompt}"
        elif self.name.lower() == "cohere":
            response = f"[Cohere-{self.model}] Processing with Command: {prompt}"
        elif self.name.lower() == "cerebras":
            client = Cerebras(
                api_key=os.environ.get("CEREBRAS_API_KEY"),  # This is the default and can be omitted
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model
            )
            try:
                response = chat_completion.choices[0].message.content
                debug_print(DEBUG_INFO, f"Successfully extracted content from Cerebras response")
            except (AttributeError, IndexError):
                debug_print(DEBUG_INFO, f"Failed to extract content from Cerebras response")
                response = f"[cerebras-{self.model}] Error processing prompt"
        else:
            response = f"[{self.name}-{self.model}] Response to prompt: {prompt}"
        
        return response

class LLMContainer:
    """
    Stores and retrieves multiple LLM clients by name/model.
    """
    def __init__(self):
        self._clients: Dict[str, LLMClient] = {}
        debug_print(DEBUG_VERBOSE, "Initialized an empty LLMContainer")

    def register_llm(self, llm_name: str, model_name: str) -> None:
        # Load provider-specific configurations
        if llm_name.lower() == "openai":
            debug_print(DEBUG_INFO, "Loading OpenAI API key...")
            product_key = self._load_api_key("OPENAI_API_KEY")
        elif llm_name.lower() == "anthropic":
            debug_print(DEBUG_INFO, "Loading Anthropic API key...")
            product_key = self._load_api_key("ANTHROPIC_API_KEY")
        elif llm_name.lower() == "cohere":
            debug_print(DEBUG_INFO, "Loading Cohere API key...")
            product_key = self._load_api_key("COHERE_API_KEY")
        elif llm_name.lower() == "cerebras":
            debug_print(DEBUG_INFO, "Loading Cerebras API key...")
            product_key = self._load_api_key("CEREBRAS_API_KEY")
        else:
            debug_print(DEBUG_INFO, f"No specific configuration for {llm_name}")
            product_key = None

        client = LLMClient(name=llm_name, model=model_name)
        key = self._make_key(llm_name, model_name)
        self._clients[key] = client
        debug_print(DEBUG_INFO, f"Registered LLM: {key}")

    def get_llm(self, llm_name: str, model_name: str) -> LLMClient:
        key = self._make_key(llm_name, model_name)
        debug_print(DEBUG_VERBOSE, f"Retrieving LLM: {key}")
        return self._clients[key]

    def _make_key(self, llm_name: str, model_name: str) -> str:
        return f"{llm_name}:{model_name}"

    def _load_api_key(self, env_var_name: str) -> str:
        """Load API key from environment variable."""
        import os
        key = os.getenv(env_var_name)
        if not key:
            debug_print(DEBUG_INFO, f"Warning: {env_var_name} not found in environment variables")
        return key
