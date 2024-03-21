# LLM Action Evaluation

# Installation
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

# Defining evaluation with Hydra
## Token 
A config file structure tree looks like the following:
``` example file structure
/config
/<Name of dataset/project>
  /<type>
    /comparator.yaml
    /metrics.yaml
  <Your_Config.yaml>
```
A corresponding hydra `<Your_Config.yaml>` file will be:
```
defaults:
  - base
  
format: csv/HF
dataset: <Name of dataset/project (e.g. Opentable)>
type: <type (e.g. trajectory)>
```
* `metrics.yaml`

```
<name of metrics>:
  on: <field of interest (e.g. column in dataset)>
### such as
bleu:
  on: explanation
```
* ` comparator.yaml`
You can add some publicly available model that're readily available to compare to your model's performance.
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


