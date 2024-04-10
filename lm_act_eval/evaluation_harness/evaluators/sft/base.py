from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import wandb

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

# from .utils import generate_completions
# from ..metrics import Action as ActionEvaluator

# from lm_act_eval.evaluation_harness.constants import CACHE_DIR
# from lm_act_eval.evaluation_harness.evaluators.utils import _validate_and_login
# from lm_act_eval.evaluation_harness.evaluators.registry import Registry
# from lm_act_eval.evaluation_harness.evaluators.common import metric_registry

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
    @abstractmethod
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


