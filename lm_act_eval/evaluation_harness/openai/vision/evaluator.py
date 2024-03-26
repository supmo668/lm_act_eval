from gptv import GPTV
import pandas as pd
from typing import *

from config import gptv_config
from .prompts import DEFAULT_EVAL_PROMPT, GPTV_EVAL_PROMPTS

from lm_act_eval.evaluation_harness.helper_functions.multion import (
  action_prefix,
  clean_extracted_text,
  extract_thought,
  extract_action,
  extract_explanation,
  ParseChatCompletion
)

class WebNavEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.initialize_fields()
    
    def initialize_fields(self):
      self.groundtruth = self.df.ground_truth, 
      self.chat_completions = self.df.chat_completion_messages
      self.screenshots= self.df.screenshot
      self.user_inputs = self.df.inputs
      self.eval_instruct = self.eval_prompt()
    @property
    def eval_prompt(self):
        # Custom logic to generate a dynamic prompt based on the given objective
        return DEFAULT_EVAL_PROMPT
    
    def process(self):
        self.eval_input_df = pd.DataFrame()
        self.eval_input_df["user_query"] = self.user_inputs.apply(
          lambda s: ParseChatCompletion().parse_as_json(
            s, target_field=None).get('QUERY',''))
        self.eval_input_df["goal"] = self.chat_completions.apply(
          lambda s: ParseChatCompletion().parse_as_json(s, target_field='content')
          ).apply(extract_thought)
      
    def synthesize_evaluation_prompts(self, instruct_texts: List):
      eval_query = pd.Series([
        DEFAULT_EVAL_PROMPT.format(GOAL=a, QUERY=b) for a, b in zip(*[
          goal_texts, user_inputs_processed])
        ])
    def evaluate_navigation(self, objective, images):
        prompt = self.generate_evaluation_prompt(objective)
        return self.generate_completion(prompt, images, openai_sdk=True)


# Example usage
if __name__ == "__main__":
    pipeline = GPTV(gptv_config)
    evaluator = WebNavEvaluator(pipeline)
    image_sources = [
        "path/to/screenshot1.jpg",
        "path/to/screenshot2.jpg",
        # Add paths or URLs as needed
    ]
    final_objective_prompt = "Assess if the webpage navigation reached the target content: a detailed product description for 'XYZ product'."
    
    evaluation_results = evaluator.evaluate_navigation(image_sources, final_objective_prompt)
    for result in evaluation_results:
        print(result)
