from abc import ABC, abstractmethod
import pandas as pd
from datasets import Dataset
from typing import Any, Union, Literal, Dict

from lm_act_eval.evaluation_harness.helper_functions.utils import function_registry
from lm_act_eval.evaluation_harness.utils.url import is_screenshot_url_accessible

from omegaconf import OmegaConf

class BaseScorer:
    def __init__(self, config: OmegaConf, *args, **kwargs):
        """
        Initializes the class with the given config using OmegaConf.

        Parameters:
            config (OmegaConf): The configuration object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.config = config

    @abstractmethod
    def evaluate(self) -> Any:
      """
      Method to perform evaluation. Sets up input dataframe with text and images, and then applies GPTV for generation into a series of text evaluation
      """
      pass
    
    @abstractmethod
    def __call__(self, dataset: Union[Dataset, pd.DataFrame], *args: Any, **kwds: Any) -> Any:
      self.input_df = dataset
      return self.evaluate()
    
class DFTableScorer(BaseScorer):
    def __init__(self, config: OmegaConf, *args, **kwargs):
        self.config = config
        self.process_df = pd.DataFrame()
    @property
    def eval_prompt(self):
      # Custom logic to generate a dynamic prompt based on the given objective
        return ""
    
    def get_last_in_trajectory(self, df, group, idx):
      return df.loc[df.groupby(group)[idx].idxmax()]
    
    @abstractmethod
    def evaluate(self) -> Any:
      assert hasattr(self, 'result')
      
    def is_eligible(self, row):  
      # Apply the function to each URL in the column
      if self.config.evaluate_group_last:
        row 
      return True
      
    def _process(self):
      assert hasattr(self, 'input_df')
      self.process_df = self.input_df[self.input_df.apply(self.is_eligible, axis=1)]
      return 
    
    def _process_result(self, evals, *args, **kwargs):
        return evals
    
    @abstractmethod
    def __call__(self, dataset: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
      self.input_df = dataset
      self._process()
      self.evals = self.evaluate()
      return self._process_result()
    
    @property
    def evaluations(self):
      return self.evals