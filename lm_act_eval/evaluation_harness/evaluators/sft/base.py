from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests
import json
import pandas
from PIL import Image
from io import BytesIO

from datasets import load_metric
import wandb

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
import pandas as pd

from .utils import generate_completions
# from ..metrics import Action as ActionEvaluator

from lm_act_eval.evaluation_harness.constants import CACHE_DIR
from lm_act_eval.evaluation_harness.utils import validate_logging
from lm_act_eval.evaluation_harness.evaluators.registry import Registry
from lm_act_eval.evaluation_harness.evaluators.common import metric_registry

class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        config: Optional[argparse.Namespace],
        model: Optional[PreTrainedModel]=None,
        **kwargs
        ):
        self.model = model
        self.config = config
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def __call__(self, eval_dataset: Optional[Dataset | pd.DataFrame]=None) -> dict | pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(self) -> Union[dict, pd.DataFrame]:
        pass

class TableEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(config.data.path)
        self.loggers: Dict = self._validate_logging()
        # Determine eligibility on metrics level
        
    @property
    def metrics(self, metrics: Union[Dict[str, callable], Registry]={}) -> Dict:
        # Placeholder for metric function registration
        return metric_registry.get(list(self.config.metrics.keys()))
    
    def _is_eligible_entry(self, row: Union[Dict, pd.Series]) -> bool:
        # Apply the function to each URL in the column
        return True
    
    def _process_result(self, evals):
        for eval in evals:
            if type(eval)==pd.DataFrame:
                
        return evals
  
    def _process_inputs(self, df):
        return df
    
    def evaluate(self, dataset: pd.DataFrame,) -> Dict:
        result = {}
        for task_name, m_func in self.metrics:
            result[task_name] = m_func(
                **self.config.metrics.get(task_name)
            )(dataset)
        return super().evaluate()
    
    @abstractmethod
    def __call__(self, dataset: pd.DataFrame, *args, **kwargs):
        self._process_inputs()
        evals:pd.DataFrame = self.evaluate()
        return self.process_result(evals)