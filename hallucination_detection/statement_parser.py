# hallucination_detection/statement_parser.py
"""
A parser to partition input text by paragraphs or word count and extract statements.
"""

import re
from typing import List

from .debug_logger import debug_print, DEBUG_INFO
from .llm import LLMContainer

class StatementParser:
    """
    The StatementParser can:
      1. Partition a text by paragraph or by a given max word limit.
      2. Extract statements (e.g., sentences) from each partition.
    """

    def __init__(self, max_words: int = 100, split_by_paragraph: bool = True):
        """
        :param max_words: Maximum words for each chunk if not strictly using paragraphs.
        :param split_by_paragraph: If True, split based on paragraphs first.
        """
        self.max_words = max_words
        self.split_by_paragraph = split_by_paragraph
        self.llm_container = LLMContainer()
        self.llm_container.register_llm("cerebras", "llama3.1-8b")
        self.llm_container.register_llm("cerebras", "llama3.3-70b")

    def partition_text(self, text: str) -> List[str]:
        """
        Partition the text into chunks. If split_by_paragraph is True, split by blank lines.
        If not, create chunks of up to max_words words each.
        """
        debug_print(DEBUG_INFO, "Starting text partitioning...")
        text = text.strip()
        if not text:
            return []

        if self.split_by_paragraph:
            # Split by blank lines to get paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            debug_print(DEBUG_INFO, f"Found {len(paragraphs)} paragraphs.")
            # Optionally, if paragraphs exceed max_words, chunk them further
            return self._chunk_paragraphs(paragraphs)
        else:
            # Directly chunk by words
            return self._chunk_by_words(text)

    def extract_statements(self, text_partition: str) -> List[str]:
        """
        Use LLM to extract statements and resolve pronouns.
        """
        debug_print(DEBUG_INFO, f"Extracting statements from partition: '{text_partition[:50]}...'")

        llm_client = self.llm_container.get_llm("cerebras", "llama3.3-70b")
        
        prompt = f"""Only change is that resolving any pronouns by replacing them with their referents, and extract individual statements from the text.  Be careful on more than one sentence describes one single statement or a logic chain, put them in one line, but keep it as oringal as possible except pronous replacement.
        Return each statement on a new line.
        Do not add any explanations or additional text.

        Text: {text_partition}
        Statements:"""

        response = llm_client.generate_text(prompt)
        
        # Split response into individual statements
        statements = [st.strip() for st in response.split('\n') if st.strip()]
        
        debug_print(DEBUG_INFO, f"Extracted {len(statements)} statements with resolved pronouns")
        return statements

    def _chunk_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        If a paragraph has more words than max_words, chunk it.
        """
        partitions = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= self.max_words:
                partitions.append(para)
            else:
                partitions.extend(self._split_long_text(words))
        return partitions

    def _chunk_by_words(self, text: str) -> List[str]:
        words = text.split()
        return self._split_long_text(words)

    def _split_long_text(self, words: List[str]) -> List[str]:
        """
        Helper to split a list of words into multiple chunks of <= max_words length.
        """
        chunks = []
        current_chunk = []
        for w in words:
            current_chunk.append(w)
            if len(current_chunk) >= self.max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        # Add any leftover words
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
