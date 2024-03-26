
from .base import CSVEvaluator
from typing import *

from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry

from lm_act_eval.evaluation_harness.helper_functions.multion import (
  action_prefix,
  clean_extracted_text,
  extract_thought,
  extract_action,
  extract_explanation,
  ParseChatCompletion
)

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

# nan-safe functions
extract_action_fs = lambda x: extract_action(x) if type(x) == str else ""
extract_thought_fs = lambda x: extract_thought(x) if type(x) == str else ""
extract_explanation_fs = lambda x: extract_explanation(x) if type(x) == str else ""
process_fs = {
  "action": extract_action_fs,
  "thought": extract_thought_fs,
  "explanation": extract_thought_fs
}

@evaluator_registry.register("csv_trajectory")
class CSVTrajectoryEvaluator(CSVEvaluator):
    def _register_metrics(self):
        # Placeholder for metric function registration
        # Example: self.metric_registry.register("accuracy", accuracy_function)
        pass

    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True
      
    def process_inputs(self):
      
        return super().process_inputs()
