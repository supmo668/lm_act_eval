from typing import Any
import pandas
from gptv import GPTV
import pandas as pd
from typing import *
from tqdm.auto import tqdm
tqdm.pandas()

from config import gptv_config
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
from lm_act_eval.evaluation_harness.evaluators.sft.base import CSVEvaluator as PdEvaluator

class GPTVEvaluator(PdEvaluator):
    def _initialize(self, df):
      self.df = df
      self.chat_completions = self.df.chat_completion_messages
      self.screenshots= self.df.screenshot
      self.user_inputs = self.df.inputs
      # 
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
      Method to perform evaluation. Sets up input dataframe with text and images, and then applies GPT-3 model for generation.
      """
      self.input_df = pd.DataFrame(index=self.df.index) 
      self.input_df['text'] = self._synthesize_evaluation_prompts()
      self.input_df['images'] = self.screenshots
      return self.input_df.progress_apply(lambda r: self.gptv.generate_completion(**r), axis=1)
    
    def __call__(self, dataset: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
      self._initialize(dataset)
      self._process()
      return self.evaluate()

# Example usage
if __name__ == "__main__":
    evaluator = GPTVEvaluator()
    image_sources = [
        "path/to/screenshot1.jpg",
        "path/to/screenshot2.jpg",
        # Add paths or URLs as needed
    ]
    final_objective_prompt = "Assess if the webpage navigation reached the target content: a detailed product description for 'XYZ product'."
    
    evaluation_results = evaluator.evaluate_navigation(image_sources, final_objective_prompt)
    for result in evaluation_results:
        print(result)
