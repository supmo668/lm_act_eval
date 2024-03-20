from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Function
import json

from torch import Tensor


@dataclass
class Metric:
    name: Optional[str] = None
    function: Function
    
  