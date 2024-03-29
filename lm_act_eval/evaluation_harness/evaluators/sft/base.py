from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests
import json
import pandas
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
import pandas as pd

from .utils import generate_completions
# from ..metrics import Action as ActionEvaluator

from lm_act_eval.evaluation_harness.constants import CACHE_DIR
from lm_act_eval.evaluation_harness.evaluators.registry import Registry
from lm_act_eval.evaluation_harness.evaluators.common import metric_registry

class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        config: Optional[argparse.Namespace],
        eval_dataset: Optional[Dataset | pd.DataFrame]=None,
        model: Optional[PreTrainedModel]=None,
        **kwargs
        ):
        self.model = model
        self.eval_dataset = eval_dataset
        self.config = config
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def __call__(self, dataset) -> dict | pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(self) -> dict:
        pass

class CSVEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config
        self.read_df = pd.read_csv(config.data.path)     
        self.df = self.read_df[self.read_df(lambda row: self._is_entry_elilgible(row), axis=1)]

    
    @property
    def metrics(self, metrics: Union[Dict[str, callable], Registry]={}) -> Dict:
        # Placeholder for metric function registration
        return metric_registry.get(list(self.config.metrics.keys()))
    
    def _process_result(self, evals):
        return evals
  
    def _process_inputs(self, df):
        return df
    
    @abstractmethod
    def __call__(self, dataset: pd.DataFrame, *args, **kwargs):
        self._process_inputs()
        evals = self.evaluate()
        return self.process_result(evals)