from typing import *
import pandas as pd

from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry

import warnings

def extract_trajectory(
  traj_df, target_col: str='ground_truth',
  session_cols: List[str]=['session_id', 'idx_in_session']
  ) -> List:
  """
  Extracts a trajectory from a DataFrame.

  Args:
      traj_df (DataFrame): The DataFrame containing the trajectory data.
      target_col (str, optional): The name of the column containing the target values. Defaults to 'ground_truth'.
      session_cols (List[str], optional): The names of the columns representing the session information. to be sorted by.

  Returns:
      List: The extracted trajectory as a list of target values, sorted by session and index within session.
  """
  sorted_grouped_texts = (
    traj_df.sort_values(by=session_cols)
    .groupby('session_id')[target_col]
    .apply(list)
  )
  return sorted_grouped_texts

from .dataframe import DataFrameEvaluator

@evaluator_registry.register("sft.trajectory")
class TableTrajectoryEvaluator(DataFrameEvaluator):
    def __init__(self, config, *args, **kwargs):
        """
        Handles evaluation of all the metrics in the given evaluation track
        """
        super().__init__(config)
        
    def _process_result(self, evals):
        return evals
    
    def _process(self):
        return self.process_df
    
    def process_result(self, evals):
        return evals