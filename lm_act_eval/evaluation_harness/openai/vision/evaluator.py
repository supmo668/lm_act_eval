from typing import Any
from omegaconf import OmegaConf
from openai import completions
import pandas

from lm_act_eval.evaluation_harness.utils.url import is_screenshot_url_accessible
from .gptv import GPTV
import pandas as pd
from typing import *
from tqdm.auto import tqdm
import re 

from lm_act_eval.ontology.config import gptv_config
from .prompts import DEFAULT_EVAL_PROMPT, GPTV_EVAL_PROMPTS

from lm_act_eval.evaluation_harness.helper_functions.multion import (
  action_prefix,
  clean_extracted_text,
  extract_thought,
  extract_action,
  extract_explanation,
  ParseChatCompletion,
  extract_first
)


from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry
from lm_act_eval.evaluation_harness.evaluators.metrics.base import DFTableScorer

from lm_act_eval.ontology.inputs import GPTVScorerInput, Optional

@metric_registry.register('gpt-v')
class GPTVScorer(DFTableScorer):
    def __init__(self, config: OmegaConf, *args, **kwargs):
        """
            A description of the entire function, its parameters, and its return types.
        """
        super().__init__(config, *args, **kwargs)
        if not args:
          self.gptv = GPTV(gptv_config)
        else: 
          self.gptv = GPTV(**config)
        self.config = config    
    
    def _process(self):
      assert all([c in self.input_df.columns for c in self.required_cols]), f"Missing all required columns: {self.required_cols}"
      tqdm.pandas(desc='Determining entry eligibility')
      self.process_df = self.input_df[self.input_df.progress_apply(self.is_eligible, axis=1)]
    
    def is_eligible(self, row):
      return is_screenshot_url_accessible(row)
    
    @property
    def eval_prompt(self):
      # Custom logic to generate a dynamic prompt based on the given objective
      # field involved:
        # QUERY
        # GOAL
      return GPTV_EVAL_PROMPTS.get('multion_trajectory')
    
    def _synthesize_and_evaluate(self, row):
      """
      Combines prompt synthesis, model generation, and result processing for a single row.
      """
      # Synthesize evaluation prompt
      prompt = self.eval_prompt.format(**row)
      
      # Generate model completion (assuming this function takes named arguments for text and image)
      completion = self.gptv.generate_completion(
        text=prompt, images=[row.screenshot])
      
      # Process and split the completion into Score and Explanation
      score, explanation = completion.split('\n', 1)
      score = extract_first(score, 'SCORE')
      explanation = extract_first(
        explanation, 'EXPLANATION')
      return pd.Series({
        'Score': score, 
        'Explanation': explanation
        })
    
    def evaluate(self):
      tqdm.pandas(desc='Evaluating with GPT-V')
      evals = self.process_df.progress_apply(
        self._synthesize_and_evaluate, axis=1)
      return evals

    def _process_result(self, evals):
      return evals['Score'].mean()
    
    def __call__(self, dataset: Union[GPTVScorerInput, pd.DataFrame], *args: Any, **kwds: Any) -> Any:
      self.input_df = dataset
      self._process()
      self.evals = self.evaluate()
      return self._process_result(self.evals)

# Example usage
if __name__ == "__main__":
  
    df = pd.DataFrame({
      'chat_completion_messages': [
          '{"target": "Find the product XYZ description", "QUERY": "Navigate to XYZ product page"}',
          '{"target": "Check shipping options for XYZ", "QUERY": "Go to shipping information section"}'
      ],
      'screenshot': [
          'path/to/screenshot1.jpg',
          'path/to/screenshot2.jpg'
          # paths to screenshots relevant to each chat completion message
      ],
      'inputs': [
          'User navigates to the XYZ product page to find descriptions',
          'User scrolls through the product page to check shipping options'
          # descriptions of user actions for each scenario
      ]
    })

    evaluator = GPTVScorer()

    # Running the evaluator
    evaluation_results = evaluator(df)
    for result in evaluation_results:
        print(result)
