# hallucination_detection/checks/latest_news_check.py

from .base_check import BaseCheck
from ..debug_logger import debug_print, DEBUG_INFO
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
from typing import List, Dict
import re

class LatestNewsCheck(BaseCheck):
    """
    Check latest news using NewsAPI and LLM verification.
    """
    def __init__(self):
        super().__init__()
        self.llm_container.register_llm("cerebras", "llama3.3-70b")
        api_key = os.environ.get('NEWS_API_KEY')
        if not api_key:
            debug_print(DEBUG_INFO, "NEWS_API_KEY not found in environment variables")
            self.news_api = None
        else:
            self.news_api = NewsApiClient(api_key=api_key)
        
    def _search_news(self, text: str) -> List[Dict]:
        """Search recent news articles related to the statement."""
        if not self.news_api:
            debug_print(DEBUG_INFO, "NewsAPI client not initialized")
            return []

        # Extract key terms from text
        if not isinstance(text, str):
            debug_print(DEBUG_INFO, f"Invalid input type: {type(text)}")
            return []

        keywords = re.sub(r'[^\w\s]', '', text).split()
        keywords = [w for w in keywords if len(w) > 3]  # Filter short words
        
        if not keywords:
            debug_print(DEBUG_INFO, "No valid keywords extracted")
            return []

        # Get news from last 7 days
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        try:
            query = ' OR '.join(keywords)
            debug_print(DEBUG_INFO, f"Searching news with query: {query}")
            
            response = self.news_api.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=5
            )
            
            articles = response.get('articles', [])
            debug_print(DEBUG_INFO, f"Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            debug_print(DEBUG_INFO, f"NewsAPI search failed: {str(e)}")
            return []

    def check_fact(self, text: str) -> float:
        debug_print(DEBUG_INFO, f"[LatestNewsCheck] Checking latest news: {text}")
        
        # First search news articles
        articles = self._search_news(text)
        
        if not articles:
            debug_print(DEBUG_INFO, "No relevant news articles found")
            return 0.1
        
        # Prepare context for LLM verification
        context = "\n".join([
            f"Headline: {a['title']}"
            for a in articles[:3]  # Use top 3 articles
        ])
        
        prompt_template = """Given these recent news headlines:
        {context}
        
        Analyze if the following statement is consistent with current news.
        Rate from 0 (completely inconsistent) to 1 (completely consistent).
        Only respond with a number between 0 and 1, nothing else.
        If unsure, give a lower score.
        
        Statement: {text}
        Consistency score:"""
        
        # Use LLM to verify statement against news context
        score = self.get_llm_truth_score(
            text, 
            prompt_template.format(context=context, text=text),
            "cerebras", "llama3.3-70b"
        )
        
        debug_print(DEBUG_INFO, f"[LatestNewsCheck] Score for '{text}': {score}")
        return score
