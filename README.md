# LLM Action Evaluation (lm_act_eval)

`lm_act_eval` is a library designed for evaluating language model actions through a variety of metrics and comparators. It leverages Hydra for flexible configuration management.

## Installation

Ensure you have Python installed on your system. Then, clone the repository and navigate to the root directory of the project to install `lm_act_eval` and its dependencies.

```bash
pip install -e .
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

Hydra allows for modular and flexible configuration of evaluations. An example configuration directory structure is provided below:

```plaintext
<ENTRY_CONFIG>.yaml
<Name of project>
eval_configuration.yaml
type/
comparator.yaml
dataset.yaml
metrics.yaml
```

### Configuration Example

A sample Hydra configuration file structure for (`<your_entry_configuration>.yaml`) might look like this:

```yaml
├── <name of the project (e.g. OpenTable)>
│   ├── default.yaml
│   └── <name of the mode (e.g. sft)>
│       ├── _group_.yaml
│       └── default.yaml
├── <your_entry_configuration>.yaml
```

#### Detailed Configuration Components

**`<your_entry_configuration>.yaml`**
specifies the following is a `eval` task and in the settings speicify eval type to be `sft`

```yaml
# config.yaml
defaults:
  - _self_
  - opentable/default@eval
# opentable/default.yml
defaults:
  - _self_
  - sft: default
```
In the `opentable/sft/default.yml` you may specify the detail of the data and evaluation settings 

```yaml
trajectory: 
  args: 
    path: .cache/five-star-trajectories/csv/data+gptv.csv
    columns:
      y: ground_truth
      y_': GPTV response
      # ... other important input fields
      extract_fs:
      - action: extract_action
      - thought: extract_thought
      - explanation: extract_explanation

  comparator:
    gptv:
      model: gpt-4-vision-preview
      max_token: 300
      img_fidelity: high
      
  metrics:
    edit_distance:
      on: actions
      
    bleu:
      on: explanation
      
    llm_relevancy:
      on: explanation
```

## Current Support

### Evaluation Data Formats

1. Pre-generated DataFrames/datasets.
2. Hugging Face (HF) models repository
### Metrics

Levenshtein Distance (AutoEval)
BLEU score
Contextual Precision test (DeepEval)

### Comparators/Benchmark Available

GPT-V