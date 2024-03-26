from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Function
import json

from torch import Tensor

@dataclass
class AgentAction:
  text: str
  goal_text: str