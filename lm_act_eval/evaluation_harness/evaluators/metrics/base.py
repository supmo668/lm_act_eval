from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Union

class DataFrameEvaluator(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.df = None
        self.process_df = pd.DataFrame()
        self.input_df = pd.DataFrame()

    @abstractmethod
    def _initialize(self, df: pd.DataFrame):
        """
        Initializes the evaluator with the dataframe.
        Subclasses should implement this to setup any necessary data from the dataframe.
        """
        pass

    @abstractmethod
    def _process(self):
        """
        Processes the dataframe in preparation for evaluation.
        Subclasses should implement the logic to transform or prepare the data.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Any:
        """
        Performs the evaluation.
        Subclasses should implement the evaluation logic and return the results.
        """
        pass

    def __call__(self, dataset: Union[pd.DataFrame], *args: Any, **kwds: Any) -> Any:
        self._initialize(dataset)
        self._process()
        return self.evaluate()