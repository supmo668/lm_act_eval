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
from ..metrics import Action as ActionEvaluator

from . import evaluator_registry
    
@evaluator_registry.register("finetuning")
class FinetuneEvaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        eval_dataset: Dataset,
        evaluate_config: argparse.Namespace,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset
        self.evaluate_config = evaluate_config

    def __call__(self, prediction: EvalPrediction) -> dict:
        generation_samples = self.evaluate_config.full_generation_samples
        prompts = self.eval_dataset["prompt"][:generation_samples]
        completions = generate_completions(
            self.tokenizer, self.model, prompts)
        pairs = list(zip(self.eval_dataset["action"], completions))
        data = [
            [prompt, action_label, generated_text]
            for prompt, (action_label, generated_text) in zip(prompts, pairs)
        ]
        wandb.log(
            {
                "generated_examples": wandb.Table(
                    data=data, columns=["Prompt", "Action Label", "Generated Text"]
                )
            }
        )

        metrics = self.compute_metrics(pairs)
        return metrics

    def compute_metrics(self, pairs: list[tuple[str, str]]) -> dict:
        evaluators = [ActionEvaluator(pred, label) for pred, label in pairs]
        metrics = {
            "right_action": np.mean([e.is_right_action() for e in evaluators]),
            "right_first_command": np.mean(
                [e.is_right_first_command() for e in evaluators]
            ),
            "right_status": np.mean([e.is_right_status() for e in evaluators]),
        }
        return metrics

from omegaconf import OmegaConf
from ..metrics import metric_registry
from agent_data.agent_data.image_utils import is_screenshot_url_accessible

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
    def __init__(self, config_path):
        self.config = OmegaConf.load(config_path)
        self._register_metrics()

    def _register_metrics(self):
        # Placeholder for metric function registration
        # Example: self.metric_registry.register("accuracy", accuracy_function)
        pass

    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True