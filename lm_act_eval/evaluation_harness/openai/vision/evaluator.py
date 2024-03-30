from typing import Any
from omegaconf import OmegaConf
import pandas
from .gptv import GPTV
import pandas as pd
from typing import *
from tqdm.auto import tqdm
import re 

tqdm.pandas()

from .config import gptv_config
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


from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry
from lm_act_eval.evaluation_harness.evaluators.metrics.base import DataFrameEvaluator as DFEvaluator

@evaluator_registry.register('GPT-V')
class GPTVEvaluator(DFEvaluator):
    def __init__(self, config: OmegaConf, *args, **kwargs):
       """
           A description of the entire function, its parameters, and its return types.
       """
       super().__init__(config, *args, **kwargs)
       self.gptv = GPTV(gptv_config)
    
    @property
    def eval_prompt(self):
      # Custom logic to generate a dynamic prompt based on the given objective
        return DEFAULT_EVAL_PROMPT
    
    def _process(self):
      self.process_df = pd.DataFrame(index=self.df.index) 
      # Using as evaluator
      self.process_df['QUERY'] = self.user_inputs.apply(lambda s: ParseChatCompletion().parse_as_json(
        s, target_field=None).get('QUERY',''))
      self.process_df['GOAL'] = self.chat_completions.apply(lambda s: ParseChatCompletion().parse_as_completion_content(s))
      
    def _synthesize_evaluation_prompts(self):
      """
      Synthesizes evaluation prompts based on the input dataframe, applying the evaluation prompt format to each row.
      """
      return self.process_df.apply(lambda r: self.eval_prompt.format(**r), axis=1)

    def evaluate(self):
      """
      Method to perform evaluation. Sets up input dataframe with text and images, and then applies GPTV for generation into a series of text evaluation
      """
      self.input_df = pd.DataFrame(index=self.df.index) 
      self.input_df['text'] = self._synthesize_evaluation_prompts()
      self.input_df['images'] = self.screenshots
      return self.input_df.progress_apply(lambda r: self.gptv.generate_completion(**r), axis=1)
    
    def _process_result(self, evals):
      result = evals.str.split('\n', n=1, expand=True).rename_axis(index=None)
      result.columns = ['Score', 'Explanation']
      # 
      result['Score'] = result['Score'].apply(lambda s: extract_first(s, term='SCORE'))
      extract_explanation = lambda x: re.search(r'EXPLANATION:\n(.+)', x, re.DOTALL).group(1) if re.search(r'EXPLANATION:\n', x) else None
      result['Explanation'] = result['Explanation'].apply(lambda s: extract_explanation(s))
      return result
    
    def __call__(self, dataset: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
      self.input_df = dataset
      self._process()
      evals = self.evaluate()
      return self._process_result(evals)

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

    evaluator = GPTVEvaluator()

    # Running the evaluator
    evaluation_results = evaluator(df)
    for result in evaluation_results:
        print(result)
