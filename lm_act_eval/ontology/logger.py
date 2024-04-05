from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class EvaluationLogs:
    input: str
    output: str
    expected: str
    scores: Dict[str, float] = field(default_factory=dict)
