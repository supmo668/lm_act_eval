from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests
import json
import pandas
from PIL import Image
from io import BytesIO
from beartype import beartype
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
from lm_act_eval.evaluation_harness.evaluators.utils import _validate_and_login
from lm_act_eval.evaluation_harness.evaluators.registry import Registry
from lm_act_eval.evaluation_harness.evaluators.common import metric_registry

from .data import cfg_to_function
import logging

logger = logging.getLogger(__name__)

class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        config: Optional[argparse.Namespace],
        model: Optional[PreTrainedModel]=None,
        **kwargs
        ):
        """
        Initializes the class with the given configuration and optional model.

        Args:
            config (Optional[argparse.Namespace]): The configuration namespace.
            model (Optional[PreTrainedModel]): The pre-trained model. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model = model
        self.config = config
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def __call__(self, eval_dataset: Optional[Dataset | pd.DataFrame]=None) -> dict | pd.DataFrame:
        pass
    
    def _process_input(self, input_data:Optional[Dataset | pd.DataFrame]=None):
        return None
    def evaluate(self) -> Union[dict, pd.DataFrame]:
        pass
    
    def log_artifacts(self):
        """Create and log a wandb db artifact or table object."""
        # Placeholder for logging logic
        # Here you would create a wandb.Table or other artifact based on self.results
        # For example, if results is a dictionary:
        log_config = self.config.logging
        pass

class DataFrameEvaluator(BaseEvaluator):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.input_df = pd.read_csv(config.path, index_col=0)
        self._process_inputs(self.input_df)

    @property
    def metric_configs(self):
        return self.config.metrics
        
    def _process_inputs(self, input_df):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        """
        self.df = pd.DataFrame()
        for (src_field, tgt_field), extract_function in cfg_to_function(self.config.extract_fs):
            logger.info(f"Extracting to {tgt_field} from {src_field}")
            self.df[tgt_field] = self.input_df[src_field].apply(extract_function)
        
    
    def process_result(self, evals):
        self.results = evals

    def evaluate(self):
        for metric, args in self.metric_configs.items():
            mfunct = self.metric_registry.get(metric)
        
    def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        self._process_inputs(input)
        evals = self.evaluate()
        return self.process_result(evals)
    
    @abstractmethod
    def log_artifacts(self):
        """Create and log a wandb db artifact or table object."""
        log_config = self.config.logging
        if self.results is not None:
            table = wandb.Table(data=[list(self.results.values())], columns=list(self.results.keys()))
            self.artifact = wandb.log({"evaluation_results": table})
        else:
            print("No results to log.")