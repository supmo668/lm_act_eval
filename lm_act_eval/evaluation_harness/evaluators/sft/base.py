from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
import json
from PIL import Image
from io import BytesIO

from datasets import load_metric


from tqdm import tqdm
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import argparse
import numpy as np
from datasets import Dataset
import wandb

from .utils import generate_completions
# from ..metrics import Action as ActionEvaluator

from lm_act_eval.evaluation_harness.constants import CACHE_DIR


class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        eval_dataset: Dataset,
        config: Optional[argparse.Namespace]=None,
        model: Optional[PreTrainedModel]=None,
        **kwargs
        ):
        self.model = model
        self.eval_dataset = eval_dataset
        self.config = config
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def __call__(self, dataset: EvalPrediction) -> dict:
        self.eval_dataset = dataset
    
    @abstractmethod
    def evaluate(self, pairs: list[tuple[str, str]]) -> dict:
        pass

class CSVEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config
        self.input_df = pd.read_csv(
            
        )
        self._get_metrics()

    def _get_metrics(self, metrics: Dict[str, callable]) -> Dict:
        # Placeholder for metric function registration
        return {}
    
    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True
    
    def process_inputs(self):
        return 