

# Metrics
## OpenTable
### Usage
To utilize the opentable_reservation_html metric, you need to initialize it with a configuration dictionary specifying the pairs of columns to be compared and any other necessary configuration settings.

Configuration Example
```
config = {
    'column_pairs': [
        {'ref': 'restaurant_name', 'pred': 'predicted_restaurant_name'},
        {'ref': 'status', 'pred': 'predicted_status'}
    ]
}
```

### Metric Initialization and Usage

`from lm_act_eval.evaluation_harness.evaluators.metrics.opentable import opentable_reservation_html`

### Initialize the metric
```
metric = opentable_reservation_html(config)
```

### Sample Input
```
data = {
    'DOM': [html_content1, html_content2],  # html_content* are strings containing HTML data
    'predicted_restaurant_name': ['Cafe Del Sol', 'Sunset Grill'],
    'predicted_status': ['confirmed', 'cancelled']
}
df = pd.DataFrame(data)

# Evaluate the metric
results = metric(df)
print(results)
```

### Output Format
The metric 
* The `__call__` metric returns a single float value

class property  `evals` 

returns a dictionary where each key corresponds to a pair of columns specified in the configuration and each value is the mean accuracy of exact matches between reference data and predicted data across the dataset.
```
{
    'restaurant_name_vs_predicted_restaurant_name': 0.95,
    'status_vs_predicted_status': 0.90
}
```

# LM Relevancy & Contextual Precision
### `external.py`
```
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
```
# Trajectory Levenshtein Distances
```
from autoevals.string import Levenshtein
```


# Other Fundamental Evaluators
* String
* Numeric

...
etc