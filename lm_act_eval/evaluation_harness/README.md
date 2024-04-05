
# Integrating DataFrame Evaluations with `lm_act_eval`

The `lm_act_eval` library offers a structured approach to evaluate language model actions, providing a versatile environment for applying a wide range of metrics and comparators. This documentation elaborates on how to integrate DataFrame-based evaluations into the Hugging Face Trainer workflow, leveraging the `lm_act_eval` library’s capabilities.

## Pre-requisite

Ensure `lm_act_eval` and its dependencies are installed as per the main README instructions. Configuration should be set according to your project’s needs, with emphasis on the evaluation configuration managed by Hydra.

## Custom Evaluation with DataFrames

### Step 1: Configure Evaluation Settings

Define your evaluation settings within a Hydra configuration file. Ensure your configuration specifies the type of evaluation (`sft` for supervised fine-tuning evaluations) and details about the dataset, including its local storage path and the relevant columns for evaluation.

### Step 2: Implement a DataFrame-based `compute_metrics` Function

Custom evaluators can be seamlessly integrated within the Hugging Face Trainer’s evaluation loop through the `compute_metrics` function. This function allows for flexibility in evaluation methodologies, including the utilization of DataFrame-based evaluations.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from lm_act_eval.config_loader import load_config
from lm_act_eval.evaluation_harness.handlers import handle_sft

def compute_metrics(eval_pred):
logits, labels = eval_pred
predictions = np.argmax(logits, axis=-1)

# Convert predictions and labels into a DataFrame
eval_df = pd.DataFrame({'predictions': predictions, 'labels': labels})

# Load Hydra configuration specific for 'sft' evaluations
cfg = load_config(config_path="config", config_name="evaluation")

# Leverage \`handle_sft\` for DataFrame-based evaluations
evaluation_results = handle_sft(cfg.eval.sft, eval_df)

# Merge custom evaluation results with traditional metrics
return {**evaluation_results}
```

### Step 3: Train and Evaluate with Custom Metrics

Integrate the `compute_metrics` function into the Hugging Face Trainer to conduct training and evaluation. This integration ensures that your custom DataFrame-based evaluation is invoked during the model’s evaluation phase.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
output_dir="./results",
num_train_epochs=3,
per_device_train_batch_size=16,
per_device_eval_batch_size=64,
warmup_steps=500,
weight_decay=0.01,
evaluate_during_training=True,
logging_dir="./logs",
)

trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=eval_dataset, # Replace with your evaluation dataset
compute_metrics=compute_metrics
)

trainer.train()
```

### Direct Evaluation with `traj_evaluation`

For cases where you want to directly apply the evaluation logic to a dataset, `lm_act_eval` allows for direct invocation via the `call` method of the evaluation classes. Here’s an example using `traj_evaluation`:

```python
from lm_act_eval.evaluation_harness.evaluators import TrajEvaluator

# Assuming `dataset` is a Pandas DataFrame with the necessary columns for evaluation
cfg = load_config(config_path=“config”, config_name=“evaluation”)
evaluator = TrajEvaluator(cfg.eval.sft)

# Directly invoke the evaluator on the dataset
evaluation_results = evaluator(dataset)
```

## Conclusion

The integration of DataFrame-based evaluations within the Hugging Face Trainer, supported by `lm_act_eval`, enriches the evaluation phase with custom metrics and detailed analysis. By following the steps outlined above, developers can leverage `lm_act_eval`'s structured approach to evaluate language model actions, applying a comprehensive set of metrics and comparators tailored to their specific evaluation needs.