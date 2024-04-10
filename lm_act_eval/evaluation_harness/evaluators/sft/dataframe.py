from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio

import wandb

from .base import BaseEvaluator
import wandb
# from tqdm import tqdm
import pandas as pd
from tqdm.asyncio import tqdm

from .utils import cfg_to_evaluator, cfg_to_function

import logging

tqdm.pandas()
logger = logging.getLogger(__name__)

async def apply_function_async(data, func):
    return func(data)

class AsyncDataFrameEvaluator(BaseEvaluator):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config

    @classmethod
    async def create(cls, config: dict):
        instance = cls(config)
        instance.input_df = pd.read_csv(config, index_col=0)
        await instance._process_inputs(instance.input_df)
        return instance
    
    @property
    def metric_configs(self):
        return self.config.metrics

    async def _process_input(self, input_df):
        """
        Processes the dataframe in preparation for evaluation using asynchronous function application.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing the data to be processed.
        """
        config_funcs = cfg_to_function(self.config['data']['extract_fs'])
        tasks = []
        
        for (src_field, tgt_field), extract_function in config_funcs:
            task = asyncio.create_task(apply_function_async(input_df[src_field], extract_function))
            tasks.append((task, src_field, tgt_field))

        await tqdm.gather(
            *[t[0] for t in tasks], 
            desc="Mapping fields to processing functions")

        for task, src_field, tgt_field in tasks:
            extracted_data = task.result()
            if isinstance(extracted_data, pd.Series):
                new_columns = {f'{tgt_field}_{col}': extracted_data[col] for col in extracted_data.index}
                self.df = pd.concat([self.df, pd.DataFrame(new_columns, index=input_df.index)], axis=1)
                logger.info(f"Multiple fields extracted to prefixed '{tgt_field}' fields")
            else:
                self.df[tgt_field] = extracted_data
                logger.info(f"Single field extracted to '{tgt_field}'")


    def apply_async(self, column_data, extract_function, tgt_field):
        results = pd.Series([extract_function(item) for item in column_data])
        if isinstance(results.iloc[0], pd.Series):
            # If multiple fields, process as a DataFrame with the extra fields
            return pd.DataFrame({f'{tgt_field}_{col}': results.apply(lambda x: x[col]) for col in results.iloc[0].index})
        else:
            # If a single field, return as a single-column DataFrame
            return pd.DataFrame({tgt_field: results})
    
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
        
        return pd.concat(concat_dfs, axis=1)

    def evaluate(self):
        self.evaluations = dict()
        for scorer_name, (scorer, inputs) in zip(self.metric_configs, cfg_to_evaluator(self.metric_configs)):   
            input_fields = [c for c in self.df.columns if c.lower().startswith(inputs)]
            self.evaluations[scorer_name] = scorer(self.df[input_fields])
        
    async def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        await self._process_input(input)
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

class AsyncDataFrameEvaluator:
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.input_df = pd.read_csv(config.data.path, index_col=0)

    @property
    def metric_configs(self):
        return self.config.metrics

    async def _process_input(self, input_df):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        
        This method updates the dataframe by applying extract functions to specified
        source fields and creating new target fields in the dataframe. If an extract
        function returns multiple fields, the names of these fields are prefixed with
        'html_' and the base target field name.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing the data to be processed.
        """
        self.df = pd.DataFrame()
        async for (src_field, tgt_field), extract_function in tqdm(
            cfg_to_function(self.config['data']['extract_fs']),
            desc="Mapping fields to processing functions"
        ):
            logger.info(f"Extracting to {tgt_field} from {src_field}")
            extracted_data = input_df[src_field].apply(extract_function)

            if isinstance(extracted_data.iloc[0], pd.Series):
                new_columns = {
                    f'{tgt_field}_{col}': await extracted_data.apply(lambda x: x[col]) for col in extracted_data.iloc[0].index
                }
                self.df = pd.concat([self.df, pd.DataFrame(new_columns)], axis=1)
                logger.info(f"Multiple fields extracted to prefixed '{tgt_field}' fields")
            else:
                self.df[tgt_field] = extracted_data
                logger.info(f"Single field extracted to '{tgt_field}'")
    
    async def apply_async(df, func):
        # Apply the async function to all elements and collect tasks
        tasks = [func(html) for html in df['HTML']]
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        # Assign results
        df['Tag_Count'] = results
        return df
    
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
            input_fields = []
            print(inputs)
            for i in inputs:
                input_fields.extend([c for c in self.df.columns if c.lower().startswith(i)])
            self.evaluations[scorer_name] = scorer(self.df[input_fields])
        
    async def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        await self._process_input(input)
        evals = self.evaluate()
        return self.process_result(evals)
          
class DataFrameEvaluator(BaseEvaluator):
    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.input_df = pd.read_csv(config.data.path, index_col=0)

    @property
    def metric_configs(self):
        return self.config.metrics

    def _process_input(self, input_df):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        
        This method updates the dataframe by applying extract functions to specified
        source fields and creating new target fields in the dataframe. If an extract
        function returns multiple fields, the names of these fields are prefixed with
        'html_' and the base target field name.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing the data to be processed.
        """
        self.df = pd.DataFrame()
        for (src_field, tgt_field), extract_function in tqdm(
            cfg_to_function(self.config.data.extract_fs),
            desc="Mapping fields to processing functions"
        ):
            logger.info(f"Extracting to {tgt_field} from {src_field}")
            extracted_data = input_df[src_field].apply(extract_function)

            # Use DataFrame structure to determine if multiple fields are returned
            if isinstance(extracted_data.iloc[0], pd.Series):
                # If multiple fields, flatten them into the main DataFrame with prefixed names
                new_columns = {f'{tgt_field}_{col}': extracted_data.apply(lambda x: x[col])
                            for col in extracted_data.iloc[0].index}
                self.df = pd.concat([
                    self.df, pd.DataFrame(new_columns)], axis=1)
                logger.info(f"Multiple fields extracted to prefixed '{tgt_field}' fields")
            else:
                # If a single field, use the target field name directly
                self.df[tgt_field] = extracted_data
                logger.info(f"Single field extracted to '{tgt_field}'")
    
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
        # Iterate over metrics to be evaluated
        for scorer_name, (scorer, inputs) in zip(self.metric_configs, cfg_to_evaluator(self.metric_configs)):   
            # use fields 
            input_fields = [c for c in self.df.columns if c.lower().startswith(inputs)]
            self.evaluations[scorer_name] = scorer(self.df[input_fields])
        
    def __call__(self, input: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        self._process_input(input)
        evals = self.evaluate()
        return self.process_result(evals)