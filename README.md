# LLM Action Evaluation

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
│       ├── \_group\_.yaml
│       └── default.yaml
├── <your\_entry\_configuration>.yaml
\```

The detailed configuration components have been enriched to accommodate a wider array of evaluation scenarios:

\```yaml
trajectory: 
  data: 
    path: lm\_act\_eval/.cache/five-star-trajectories/csv/data+gptv-eligible.csv
    columns:
      y: ground\_truth
      y\_': GPTV\_generations
    extract\_fs:
      QUERY:
        QUERY:
      screenshot:
        screenshot:
      GOAL:
        chat\_completion\_messages: parse\_completion.parse\_content
      explanation: 
        <explanation_field>: <name of function for extraction>
    logging:
      wandb:
        project: opentable
        result: lm\_act\_eval-run
      braintrust:
        project: multion\_opentable
  comparator:
    gpt-v:
      model: gpt-4-vision-preview
      max\_token: 300
      img\_fidelity: high
  metrics:
    gpt-v:
      inputs:
        - GOAL
        - QUERY
        - screenshot
    llm\_relevancy:
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