# hallucination_detection/checks/paper_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO
from scholarly import scholarly
import arxiv
import re

class PaperCheck(BaseCheck):
    """
    Check for academic paper references using Google Scholar and arXiv.
    """
    def __init__(self):
        self.scholar_client = scholarly
        self.arxiv_client = arxiv.Client()

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[PaperCheck] Checking paper fact: {text}")
        
        # Extract potential paper title using simple heuristic
        title_match = re.search(r'"([^"]*)"', text) or re.search(r"'([^']*)'", text)
        if title_match:
            paper_title = title_match.group(1)
        else:
            paper_title = text
            
        debug_print(DEBUG_INFO, f"Searching for paper: {paper_title}")
        
        # Try Google Scholar first
        try:
            search_query = scholarly.search_pubs(paper_title)
            first_result = next(search_query, None)
            if first_result:
                debug_print(DEBUG_INFO, "Found paper in Google Scholar")
                return 0.95
        except Exception as e:
            debug_print(DEBUG_INFO, f"Google Scholar search failed: {e}")

        # Try arXiv if Google Scholar fails
        try:
            search = arxiv.Search(
                query=paper_title,
                max_results=1
            )
            results = list(self.arxiv_client.results(search))
            if results:
                debug_print(DEBUG_INFO, "Found paper in arXiv")
                return 0.90
        except Exception as e:
            debug_print(DEBUG_INFO, f"arXiv search failed: {e}")

        debug_print(DEBUG_INFO, "Paper not found in either database")
        return 0.1  # Low score if paper not found
