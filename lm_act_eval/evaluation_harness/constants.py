CACHE_DIR = ".cache"

from datetime import datetime

DATETIME_STR = datetime.now().strftime("%m%d%YT%H%M%S")

import torch
from transformers import BitsAndBytesConfig

from typing import Dict

BNB_QUANTIZATION_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

BNB_QUANTIZATION_CONFIG_8BIT = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="int4",
    bnb_8bit_compute_dtype=torch.float16,
)

class BNB_OFFLOADING_CONFIG:
    quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)
    low_cpu_mem_usage: bool=True
    device_map: str ="auto"
    max_memory: Dict ={0: '80000Mib', "cpu": '16Gib'}
    
    
#### Logging
LOGGING_PLATFORMS = {
  'wandb': 'WANDB_API_KEY', 
  'hf': 'HUGGINGFACE_API_KEY'
}

