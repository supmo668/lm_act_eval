from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import wandb

from .base import BaseEvaluator
import wandb
import pandas as pd

from .utils import cfg_to_evaluator, cfg_to_function

import logging

logger = logging.getLogger(__name__)

class DataFrameEvaluator(BaseEvaluator):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.input_df = pd.read_csv(config.data.path, index_col=0)
        self._process_inputs(self.input_df)


    @property
    def metric_configs(self):
        return self.config.metrics

    def _process_inputs(self, input_df):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        """
        self.df = pd.DataFrame()
        for (src_field, tgt_field), extract_function in cfg_to_function(self.config.data.extract_fs):
            logger.info(f"Extracting to {tgt_field} from {src_field}")
            self.df[tgt_field] = self.input_df[src_field].apply(extract_function)
    
    def process_result(self):
        concat_dfs = []
        # Iterate over the dictionary items
        for names, df in self.evaluations.items():
            # Ensure the DataFrame index is a default RangeIndex for proper alignment
            df = df.reset_index(drop=True)
            # Create a MultiIndex for the columns with the names as the top level
            # and the original columns as the second level
            df.columns = pd.MultiIndex.from_product(
                [[names], df.columns])
            # Append the modified DataFrame to the list
            concat_dfs.append(df)
        
        # Concatenate all DataFrames along the columns
        # This automatically handles different lengths and merges by index
        composite_df = pd.concat(concat_dfs, axis=1)

    def evaluate(self):
        self.evaluations = dict()
        for scorer_name, (scorer, inputs) in zip(self.metric_configs, cfg_to_evaluator(self.metric_configs)):
            self.evaluations[scorer_name] = scorer(self.df[inputs])
        
    def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        self._process_inputs(input)
        evals = self.evaluate()
        return self.process_result(evals)
    
    @abstractmethod
    def log_artifacts(self):
        """Create and log a wandb db artifact or table object."""
        log_config = self.config.data.logging
        if self.results is not None:
            table = wandb.Table(data=[list(self.results.values())], columns=list(self.results.keys()))
            self.artifact = wandb.log({"evaluation_results": table})
        else:
            print("No results to log.")
          