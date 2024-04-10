import dataclasses
from typing import List, Literal, Dict

@dataclasses.dataclass
class gptv_config:
  model: str = "gpt-4-vision-preview"
  max_tokens: int = 300
  img_fidelity: str = "high"
