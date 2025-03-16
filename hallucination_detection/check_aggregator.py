# hallucination_detection/check_aggregator.py

from typing import Dict

from .domain_classification import DomainClassifier
from .checks.base_check import BaseCheck
from .checks.history_check import HistoryCheck
from .checks.paper_check import PaperCheck
from .checks.math_check import MathCheck
from .checks.logic_check import LogicCheck
from .checks.general_check import GeneralCheck
from .checks.general_check import NoneCheck
from .checks.latest_news_check import LatestNewsCheck
from .debug_logger import debug_print, DEBUG_INFO

class CheckAggregator:
    """
    Aggregates multiple domain checks. Chooses the correct check based on domain classification.
    """

    def __init__(self):
        # Pre-instantiate or lazy-load checks as needed
        self.check_map: Dict[str, BaseCheck] = {
            "history": HistoryCheck(),
            "paper": PaperCheck(),
            "math": MathCheck(),
            "logic": LogicCheck(),
            "general": GeneralCheck(),
            "latest_news": LatestNewsCheck(),
            "none": NoneCheck(),
        }
        self.domain_classifier = DomainClassifier()

    def check_statement(self, text: str) -> tuple[float, str]:
        debug_print(DEBUG_INFO, f"Aggregator is about to classify and check: {text}")
        domain = self.domain_classifier.classify(text)
        checker = self.check_map.get(domain, GeneralCheck())
        debug_print(DEBUG_INFO, f"Domain classified as '{domain}'. Using '{checker.__class__.__name__}'")
        score = checker.check_fact(text)
        return score, domain
