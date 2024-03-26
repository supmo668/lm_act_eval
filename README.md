# LLM Action Evaluation

## Installation
```
pip install -e .
```
should install all dependencies and also add `lm_act_eval` to path

To show hydra definition, run
``` 
lm_act_eval --help
```
or 
```
python -m lm_act_eval --help
```
### Set-up tokens
Set-up tokens according to `.example.env`

## Defining evaluation with Hydra

A config file structure tree looks like the following:
  ``` example file structure
  <ENTRY_CONFIG.YAML>
  <Name of project>
    <eval_configuration.yaml>
    <type>
      # stores configuration to endpoints or model to produce validation
      comparator.yaml
      # dataset path or meta information
      dataset.yaml
      # metrics to run the pipeline on
      metrics.yaml
    
  ```
A corresponding hydra `<your_entry_configuration.yaml>` file will be:
```
defaults:
  - base
  - dataset@eval_conf: dataset
  - metrics@eval_conf: trajectory
  - comparator@eval_conf: default
  
format: csv/HF
dataset: <Name of project (e.g. Opentable)>
type: <type (e.g. trajectory)>
```
##
* `dataset.yaml`
Involve dataset path or other metadata, and extracting elements/features for downstream comparison, such as:
```
path: <path to dataset>
columns:
  y: <name of groundtruth field>
  y_prime: <name of target, y'>
  extract_fs:
  - <extracted field name>: <name of extract function>
```
  - example
    ```
    path: .cache/five-star-trajectories/csv/data+gptv.csv
    columns:
      y: ground_truth
      y_prime: GPTV response
      extract_fs:
      - action: extract_action
      - thought: extract_thought
      - explanation: extract_explanation
      # None: no change
      - ground_truth_: 
    ```

* `metrics.yaml`
  
  Metrics to use
  ```
  <name of metrics>:
    on: <extracted field of interest (i.e. column or feature in dataset)>
  ```
  - example
    ```
    bleu:
      on: explanation
    ```


* ` comparator.yaml`

  You can add some publicly available model that're readily available to compare to your model's performance.

  - example 
    ```
    gptv:
      model: gpt-4-vision-preview
      max_token: 300
      img_fidelity: high
    ```
# Current Support
## Evaluation data format

* Pre-generated dataframe/dataset

* HF models

## Metrics

* Levinshtein Distance (AutoEval)
* BLEU score 
* Relevancy test (DeepEval)

## Comparators available

* GPT-V


