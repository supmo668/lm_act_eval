from lm_act_eval.evaluation_harness.registry import Registry
from  .trajectory import TrajectoryEvaluator

metric_registry = Registry()

metric_registry.register("trajectory", TrajectoryEvaluator)