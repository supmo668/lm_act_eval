

from autoevals.llm import *
from autoevals.string import Levenshtein
import warnings

from typing import Dict, List, Tuple

from beartype import beartype

from ..webarena_rl.base import Evaluator, Trajectory

from playwright.sync_api import CDPSession, Page

from evaluation_harness.helper_functions import (
    PseudoPage,
    llm_fuzzy_match,
    llm_ua_match,
)

class NumericEvaluator(Evaluator):
    """Check if the numerical relationship is correct"""
    levenshtein_comparator = Levenshtein()
    @staticmethod
    @beartype
    def edit_distance(ref: str, pred: str) -> float:
      """
      Calculate the edit distance between two strings.

      Parameters:
        ref (str): The reference string.
        pred (str): The predicted string.

      Returns:
        float: The edit distance between the reference and predicted strings.
      """
      return NumericEvaluator.levenshtein_comparator(ref, pred).score

    def __call__(
        self,
        trajectory: Tuple[List[List], List[List]],
        metrics: list,
        page: Page | PseudoPage | None = None,
        client: CDPSession | None = None,
      ) -> Dict[str, int]:
        """
        Call method to calculate the score based on the given trajectory and metrics. 

        Args:
            action trajectory (List[Tuple[List, List]]): The action trajectory data.
            metrics (list): The list of metrics to calculate score.
            page (Page | PseudoPage | None, optional): The page type. Defaults to None.
            client (CDPSession | None, optional): The client session. Defaults to None.

        Returns:
            Dict[str, int]: The calculated score for each approach.
        """
        score = dict()
        for approach in metrics:
            match approach:
                case "exact_match":
                    score.update(
                      {approach: self.edit_distance(**trajectory)}
                    )