
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

from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry

@evaluator_registry.register("finetuning")
class FinetuneEvaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        eval_dataset: Dataset,
        evaluate_config: argparse.Namespace,
    ):
        """
        Initialize the class with the provided tokenizer, model, evaluation dataset, and evaluation configuration.
        
        Parameters:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used.
            model (PreTrainedModel): The model to be used.
            eval_dataset (Dataset): The evaluation dataset.
            evaluate_config (argparse.Namespace): The configuration for evaluation.
        """
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