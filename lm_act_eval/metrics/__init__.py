from ..registry import Registry

class MetricRegistry(Registry):
  def get_metrics(self, names):
        """
        Generate a dictionary of available metrics based on the input names.

        Parameters:
            names (list): A list of metric names to check against the available metrics.

        Returns:
            dict: A dictionary containing only the available metrics from the input names.
        """
        available_metrics = {}
        for name in names:
            if name in self.metrics:
                available_metrics[name] = self.metrics[name]
            else:
                print(f"Warning: Metric '{name}' is not registered and will be skipped.")
        return available_metrics
      
metric_registry = MetricRegistry()

__all__ = ["metric_registry"]