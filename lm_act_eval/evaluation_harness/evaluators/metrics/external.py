"""
Eval toolkits provided by 3rd party/opensource libraries
"""
from typing import *
from typing import Any
from autoevals.string import Levenshtein
from deepeval.metrics import ContextualPrecisionMetric
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry

levenshtein_comparator = Levenshtein()
contextual_precision = ContextualPrecisionMetric()


metric_registry.register("edit_distance")(levenshtein_comparator)

@metric_registry.register("contextual_precision")
class contextual_precision:
  def __init__(
    self, threshold:float=0.7, model:str="gpt-4", include_reason:bool=True ):
    """
    to construct the inpuut for the function:
    https://docs.confident-ai.com/docs/metrics-contextual-precision
    ```
      test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
      )
    ```
    """
    self.metric = ContextualPrecisionMetric(
      threshold=threshold, model=model,
      include_reason=include_reason
    )
    self._is_called = False
    
  def __call__(
    self, input: str, 
    actual_output: Union[str, List[str]], expected_output: Union[str, List[str]],
    retrieval_context=Union[List[str], str]) -> Any:
    # retrieveal context needs to be a list of >=1
    if isinstance(retrieval_context, str):
      retrieval_context = [retrieval_context]
    # create test case
    test_case = LLMTestCase(
      input=input, actual_output=actual_output,
      expected_output=expected_output, retrieval_context=retrieval_context
    )
    self._is_called = True
    return self.metric.measure(test_case)
  
  @property
  def score(self):
    if not self._is_called:
            raise AttributeError("You must call the object before accessing 'score'")
    return self.metric.score
  
  @property
  def explanation(self):
    if not self._is_called:
            raise AttributeError("You must call the object before accessing 'explanation'")
    return self.metric.reason