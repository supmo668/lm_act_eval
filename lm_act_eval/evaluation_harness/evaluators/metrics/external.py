"""
Eval toolkits provided by 3rd party/opensource libraries
"""
from autoevals.string import Levenshtein
from deepeval.metrics import ContextualPrecisionMetric

from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry

levenshtein_comparator = Levenshtein()
contextual_precision = ContextualPrecisionMetric(
  threshold=0.7,model="gpt-4", include_reason=True
)


metric_registry.register("edit_distance")(levenshtein_comparator)

metric_registry.register("contextual_precision")(contextual_precision)