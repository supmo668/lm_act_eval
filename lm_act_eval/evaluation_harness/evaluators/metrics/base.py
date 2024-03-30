from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Union

from 
class DataFrameEvaluator(ABC):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.process_df = pd.DataFrame()
        self.input_df = pd.DataFrame()
    
    @abstractmethod
    def _process_inputs(self):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        """
        self.process_df = self.input[
            self.input(lambda row: self._is_entry_elilgible(row), axis=1)]
        
        return self.process_df
        
    def _is_entry_elilgible(self, row):
        # Apply the function to each URL in the column
        return True
    
    def _process(self):
        return self.process_df
    
    @abstractmethod
    def evaluate(self) -> Any:
        """
        Performs the evaluation.
        Subclasses should implement the evaluation logic and return the results.
        """
        pass
    
    def process_result(self, evals):
        return evals

    def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        self._process_inputs(input)
        self._process()
        evals = self.evaluate()
        return self.process_result(evals)