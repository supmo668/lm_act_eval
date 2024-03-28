
from .base import CSVEvaluator
from typing import *
import pandas as pd

from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry

from .process import process_fs


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


@evaluator_registry.register("sft.trajectory")
class CSVTrajectoryEvaluator(CSVEvaluator):
    def _register_metrics(self):
        # Placeholder for metric function registration
        # Example: self.metric_registry.register("accuracy", accuracy_function)
        pass

    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True
      
    def process_inputs(self):
        metric_comp = pd.DataFrame()\
        # Prepare metric input columns
        for c in [gt_col, gen_col]:
          for k, func in process_fs.items():
            metric_comp[c+'_'+k] = metric_comp[c].progress_apply(func)
        return super().process_inputs()
