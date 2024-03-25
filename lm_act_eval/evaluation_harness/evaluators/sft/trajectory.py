
from .base import CSVEvaluator
from typing import *

from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry

def extract_trajectory(traj_df, target_col: str='ground_truth') -> List:
  sorted_grouped_texts = (
    traj_df.sort_values(by=['session_id', 'idx_in_session'])
    .groupby('session_id')[target_col]
    .apply(list)
  )
  return sorted_grouped_texts

@evaluator_registry.register("csv_trajectory")
class CSVTrajectoryEvaluator(CSVEvaluator):
    def _register_metrics(self):
        # Placeholder for metric function registration
        # Example: self.metric_registry.register("accuracy", accuracy_function)
        pass

    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True
      
