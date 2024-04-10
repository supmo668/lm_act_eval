from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Field
import json, re

from torch import Tensor
from lm_act_eval.evaluation_harness.evaluators.registry import Registry

@dataclass
class Metric:
    name: Optional[str] = None
    registry: Registry
    function: Callable = Field(init=False)

    def __post_init__(self):
        if self.name:
            self.function = self.registry.get(self.name)
            if self.function is None:
                raise ValueError(f"No function found in the registry for the name {self.name}")
        else:
            self.function = None  
            # or set to a default function if appropriate

@dataclass
class MetricInput:
    ground_truth: List[Any]  # The true labels or values. Generic type 'Any' to accommodate different data types.
    predictions: List[Any]  # The predicted labels or values from a model. Matches the type of ground_truth.
    additional_info: dict = field(default_factory=dict)  # Optional field for any additional data needed.

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