
from .base import CSVEvaluator
from lm_act_eval.evaluation_harness.registry import evaluator_registry

@evaluator_registry.register("trajectory")
class CSVTrajectoryEvaluator(CSVEvaluator):
    def _register_metrics(self):
        # Placeholder for metric function registration
        # Example: self.metric_registry.register("accuracy", accuracy_function)
        pass

    def _determine_eligibility(self, row):
        # Apply the function to each URL in the column
        return True