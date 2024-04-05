# LLM Action Evaluation

`lm_act_eval` is a library designed for evaluating language model actions through a variety of metrics and comparators. It leverages Hydra for flexible configuration management.

## Installation

Ensure you have Python installed on your system. Then, clone the repository and navigate to the root directory of the project to install `lm_act_eval` and its dependencies.

```bash
pip install -e .
```
or via test-pip
```
pip install -i https://test.pypi.org/simple/ lm_act_eval
```
This command installs all necessary dependencies and adds `lm_act_eval` to your system path for easy execution.

## Usage

### Displaying Help Information

To understand the available commands and their descriptions, you can use:

```bash
lm_act_eval --help
```

Or equivalently:

```bash
python -m lm_act_eval --help
```

### Configuration Tokens

Before running evaluations, ensure to set up necessary tokens and credentials as per the `.example.env` file. Rename `.example.env` to `.env` and fill in your details.

## Defining Evaluations with Hydra

Hydra allows for modular and flexible configuration of evaluations. The configuration system has been updated to provide more granular control over the evaluation process, including data sourcing, feature extraction, logging, and metrics calculation.

### Updated Configuration Structure

An example updated Hydra configuration directory structure and files would be as follows:

```plaintext
<ENTRY_CONFIG>.yaml

eval_configuration.yaml
type/
comparator.yaml
dataset.yaml
metrics.yaml
```

### Configuration Example

A sample updated Hydra configuration for defining evaluations (`<your_entry_configuration>.yaml`) might include:

```yaml
├── <name of the project (e.g. OpenTable)>
│ ├── default.yaml
│ └── <name of the mode (e.g. sft)>
│ ├── _group_.yaml
│ └── default.yaml
├── <your_entry_configuration>.yaml
```

The detailed configuration components have been enriched to accommodate a wider array of evaluation scenarios:

```yaml
├── <name of the project (e.g. OpenTable)>
│   ├── default.yaml
│   └── <name of the mode (e.g. sft)>
│       ├── _group_.yaml
│       └── default.yaml
├── <your_entry_configuration>.yaml
```

The detailed configuration components have been enriched to accommodate a wider array of evaluation scenarios:

```yaml
trajectory: 
  data: 
    path: lm_act_eval/.cache/five-star-trajectories/csv/data+gptv-eligible.csv
    columns:
      y: ground_truth
      y_': GPTV_generations
    extract_fs:
      QUERY:
        QUERY:
      screenshot:
        screenshot:
      GOAL:
        chat_completion_messages: parse_completion.parse_content
      explanation: 
        <explanation_field>: <name of function for extraction>
    logging:
      wandb:
        project: opentable
        result: lm_act_eval-run
      braintrust:
        project: multion_opentable
  comparator:
    gpt-v:
      model: gpt-4-vision-preview
      max_token: 300
      img_fidelity: high
  metrics:
    gpt-v:
      inputs:
        - GOAL
        - QUERY
        - screenshot
    llm_relevancy:
      - explanation
```

### Current Support

#### Evaluation Data Formats

Pre-generated DataFrames/datasets.
Hugging Face (HF) models repository.
#### Metrics

Levenshtein Distance (AutoEval)
BLEU score
Contextual Precision test (DeepEval)
LLM Relevancy
#### Comparators/Benchmark Available

GPT-V


## Data Logging

The `lm_act_eval` library now includes support for comprehensive data logging, enabling you to track and store evaluation results across multiple platforms. This feature allows for better analysis and visualization of the performance of language model actions under various metrics and comparators.

### Supported Logging Platforms

**Weights & Biases (wandb)**: For real-time tracking and visualization of experiments.
**Braintrust**: For managing and sharing evaluation outcomes within a team or organization.
(Commented out in the example but supported) **Hugging Face Spaces**: For showcasing results in a publicly accessible space.
### Configuring Logging

To configure logging for your evaluations, update the `logging` section of your evaluation configuration file. Specify the logging platforms you wish to use and provide the necessary details such as project names or API keys. Below is an example configuration:

```yaml
logging:
wandb:
project: opentable
result: lm_act_eval-run
braintrust:
project: multion_opentable
# Uncomment and configure the following section to log results to Hugging Face Spaces
# hf:
# space: multion-agi
```

### Setup Instructions

**Weights & Biases**:

Sign up or log in to your Weights & Biases account.
Create a new project or use an existing one.
Obtain your API key from your account settings.
Ensure your `.env` file or environment variables include `WANDB_API_KEY` set to your Weights & Biases API key.

**Braintrust**:

Contact your Braintrust administrator to set up a project and obtain necessary credentials.
Include these details in the logging configuration under the `braintrust` section.

**(Optional) Hugging Face Spaces**:

If you want to log results to Hugging Face Spaces, ensure you have a Hugging Face account and have created a Space.
Uncomment the `hf` section in the logging configuration and provide your Space name.
Your Hugging Face API key must be stored securely and provided to the `lm_act_eval` library via environment variables or the `.env` file.