from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Union, Literal, Dict
import logging

from lm_act_eval.evaluation_harness.helper_functions.utils import function_registry

from .data import cfg_to_function

logger = logging.getLogger(__name__)

class DataFrameEvaluator(ABC):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.input_df = pd.read_csv(config.path, index_col=0)
        self.df = self._process_inputs(self.input_df)

    def _process_inputs(self, input_df):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        """
        df = input_df[input_df.apply(self._is_entry_elilgible, axis=1)]
        
        for (src_field, tgt_field), extract_function in cfg_to_function(self.config.extract_fs):
            logger.info(f"Extracting to {tgt_field} from {src_field}")
            df[tgt_field] = df[src_field].apply(extract_function)
        return df
        
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