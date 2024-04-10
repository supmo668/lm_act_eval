from cv2 import DFT_COMPLEX_INPUT
from .string import StringEvaluator
from typing import *
import pandas as pd

from .base import DFTableScorer
from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry

@metric_registry.register("opentable_html")
class opentable_reservation_html(DFTableScorer):
  """
  A class for the OpenTable metric.

  This metric evaluates the accuracy of the OpenTable dataset,
  which contains restaurant reviews.
  """
  
  def __init__(self, config):
    """
    Initialize the OpenTable metric.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """
    # Initialize the StringEvaluator with the given configuration.
    # This is used to evaluate the exact matching between the generated
    # HTML and the ground truth HTML.
    self.str_evaluator = StringEvaluator(config)

  
  def __call__(self, df: pd.DataFrame):
      """
      Evaluate the dataset by applying exact matches on configured column pairs.

      Args:
          df (pd.DataFrame): The dataset containing the actual and predicted columns.

      Returns:
          dict: A dictionary containing the results of exact matches for each column pair.
      """
      results = {}
      # Iterate over configured column pairs from self.config assumed to be provided as list of dicts
      for column_pair in self.config['column_pairs']:
          ref_col = column_pair['ref']
          pred_col = 'html_' + ref_col
          col_name = f"{ref_col}_vs_{pred_col}"
          # Using apply to call the exact_match function on each row
          results[col_name] = df.apply(
              lambda row: self.str_evaluator.exact_match(row[ref_col], row[pred_col]), axis=1
          ).mean()  # Calculating the mean to provide an overall accuracy metric for each column pair
      return results