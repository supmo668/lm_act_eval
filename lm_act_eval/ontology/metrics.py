from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import json, re

from torch import Tensor


@dataclass
class Metric:
    name: Optional[str] = None
    function: Callable
    
@dataclass
class GPTV:
    raw_text: str
    score: str = ''
    explanation: str = ''

    def __post_init__(self):
        """
        Initialize the object after it has been created by calling the extract_data method.
        """
        self.extract_data()

    def extract_data(self):
        # Pattern to match the score and explanation text
        pattern = r"Score:\s*(?P<score>[^\n]+)\nExplanation:\s*(?P<explanation>.+)"
        
        # Using DOTALL to make '.' match newlines as well
        match = re.search(pattern, self.raw_text, re.DOTALL)
        if match:
            self.score = match.group('score').strip()
            self.explanation = match.group('explanation').strip()