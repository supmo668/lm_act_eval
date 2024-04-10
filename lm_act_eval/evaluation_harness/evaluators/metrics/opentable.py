from cv2 import DFT_COMPLEX_INPUT
from .string import StringEvaluator
from typing import *
import pandas as pd

from .base import DFTableScorer
from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry
from lm_act_eval.evaluation_harness.helper_functions import function_registry

import warnings

from lm_act_eval.evaluation_harness.helper_functions import function_registry
extract_reservation_info = function_registry.get('opentable_extract_reservation_details')

@metric_registry.register("opentable_html")
class opentable_reservation_html(DFTableScorer):
    """
    A class for the OpenTable metric.

    This metric evaluates the accuracy of the OpenTable dataset,
    which contains restaurant reviews.
    """

    def __init__(self, config):
        """
        Initialize the OpenTable metric.

        Args:
            config (dict): A dictionary containing the configuration parameters.
        """
        self.config = config
        self.str_evaluator = StringEvaluator(config)

    def _process_input(self, df):
        """
        Preprocess the dataframe by extracting reservation details from the 'DOM' column.

        Args:
            df (pd.DataFrame): The dataset containing the HTML content in the 'DOM' column.

        Returns:
            pd.DataFrame: A modified dataframe with additional columns for each reservation detail.
        """
        # Apply extract_reservation_info to each row in the 'DOM' column
        if self.config.evaluate_group_last:
            df = self.get_last_in_trajectory(df, **self.config.evaluate_group_last)
        df_details = df['DOM'].apply(lambda x: extract_reservation_info(x))
        details_df = pd.DataFrame(df_details.tolist(), index=df.index)
        # Rename columns to have 'predicted_' prefix
        details_df = details_df.rename(columns=lambda x: 'predicted_' + x)
        # Concatenate the new details dataframe with the original dataframe
        df = pd.concat([df, details_df], axis=1)
        return df

    def __call__(self, df: pd.DataFrame):
        """
        Evaluate the dataset by applying exact matches on configured column pairs.

        Args:
            df (pd.DataFrame): The dataset containing the actual and predicted columns.

        Returns:
            dict: A dictionary containing the results of exact matches for each column pair.
        """
        # First process the input dataframe to extract reservation details
        df = self._process_input(df)
        
        results = {}
        # Iterate over configured column pairs from self.config assumed to be provided as list of dicts
        for column_pair in self.config['column_pairs']:
            ref_col = column_pair['ref']
            pred_col = 'predicted_'+ ref_col  # Use a predefined prediction column from the config
            if not (ref_col in df.columns or pred_col in df.columns):
                warnings.warn(f"Reference or predicted column '{ref_col}' not found in DataFrame. Skipping this pair.")
                continue  # Skip this iteration if the reference column does not exist
            col_name = f"{ref_col}_vs_{pred_col}"
            # Using apply to call the exact_match function on each row
            results[col_name] = df.apply(
                lambda row: self.str_evaluator.exact_match(row[ref_col], row[pred_col]), axis=1
            ).mean()  # Calculating the mean to provide an overall accuracy metric for each column pair
        self._score_series = pd.Series(results)
        # Calculate and return the overall average score
        return self._score_series.mean() if not self._score_series.empty else 0.0