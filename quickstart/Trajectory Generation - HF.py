# %%
from pathlib import Path
import os
from dotenv import load_dotenv


# %%
# change dir to ml repo at ml/ root (if you started in notebook/)
os.chdir('../')
print(Path.cwd())
load_dotenv()

# %%
# read dataset into datasets
from datasets import Dataset
import pandas as pd
dataset_name = "five-star-trajectories"
dataset_path = Path(f"lm_act_eval/.cache/{dataset_name }")
traj_df = pd.read_csv(dataset_path/"csv/data.csv")
traj_dataset = Dataset.from_pandas(traj_df)
print(traj_df.columns)
traj_df[traj_df.session_id==traj_df.session_id.unique()[0]]
traj_df.head(2)

# %%
from lm_act_eval.evaluation_harness.helper_functions.multion import (
  action_prefix,
  clean_extracted_text,
  extract_thought,
  extract_action,
  extract_explanation,
  ParseChatCompletion
)
from lm_act_eval.evaluation_harness.utils.url import is_screenshot_url_accessible
from typing import Callable
from tqdm import tqdm
tqdm.pandas()


# %% [markdown]
# ### Load other data

# %%
traj_df = pd.read_csv(dataset_path/'csv/data+gptv.csv')

# %% [markdown]
# ### completions

# %%
from lm_act_eval.evaluation_harness.constants import BNB_QUANTIZATION_CONFIG_8BIT, BitsAndBytesConfig

from lm_act_eval.common.hf import load_model_and_tokenizer
from lm_act_eval.evaluation_harness.evaluators.sft.utils import generate_text_and_merge

# %%
from concurrent.futures import ProcessPoolExecutor
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "MULTION-AI/feasible-morning-17-v4"
# Load the tokenizer and model
bnb = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
if not bnb:
  model = AutoModelForCausalLM.from_pretrained(model_name)
else:
  quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)
  model = AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True, 
    device_map="auto", quantization_config=quantization_config, 
    max_memory={**{k:'8000Mib' for k in range(0,8)}, "cpu": '100Gib'}
  )

# %%
traj_df_generated = generate_text_and_merge(
    traj_df,
    "chat_completion_messages_content",
    model,
    tokenizer,
    model.device,
    max_new_tokens=256,
    batch_size=2
)

# %%
traj_df_new_gen = traj_df.copy()
traj_df_new_gen.to_csv(dataset_path/f'csv/data+gen_{model_name}.csv', index=False)

# %%
# def generate_respsonse(
#     row, pipeline, inp_column="completion_msg_content", log=False, max_new_tokens=256):
#     response = pipeline(
#         row[inp_column], max_new_tokens=max_new_tokens, num_return_sequences=1)
#     generated_text = response[0]["generated_text"]
#     if log:
#         wandb.log({
#             "original_text": row[inp_column],
#             "generated_text": generated_text})
#     return generated_text

# # responses = []
# # for index, row in tqdm(traj_df.iterrows(), desc="generating completions"):
# #     responses.append(generate_respsonse(
# #         row, inp_column="completion_msg_content", log=False
# #     ))
# # traj_df['generated_responses'] = responses


# %% [markdown]
# Logging - generation

# %%
import wandb
from dotenv import load_dotenv
load_dotenv()
# Initialize Wandb
wandb.login()
run = wandb.init(
  project="trajectory_eval", entity="multion-agi",
  name=f"opentable-{model_name}", reinit=True)

# %%
opentable_artifact = wandb.Artifact(f"opentable_trajectories_eval-{model_name}", type="dataset")
# opentable_table = wandb.Table(dataframe=traj_df_new)
eval_generation = wandb.Table(dataframe=traj_df)
#
# opentable_artifact.add(opentable_table, "opentable")
opentable_artifact.add(eval_generation, "eval-generations")


# %%
# Log the table to visualize with a run...
run.log({"eval-generations": eval_generation})
# and Log as an Artifact to increase the available row limit!
run.log_artifact(opentable_artifact)

# %% [markdown]
# ### Process data

# %% [markdown]
# ### Eval Metrics

# %%
from autoevals.llm import *
from autoevals.string import Levenshtein
import warnings
levenshtein_evaluator = Levenshtein()

# %%
config_metrics = {
    "Actions": {"edit": "edit"}
}
metric_registry = {
    "edit": Levenshtein()
}
def evaluate_trajectory(dataset):
    """
    eligible dataset
    """
    metric_results = []
    for index, row in dataset[eligible].iterrows():
        result_row = {}
        for metric_config in config_metrics:
            metric_name = metric_config["name"]
            if metric_registry.get(metric_name):
                metric_func = metric_registry[metric_name]
                result_row[metric_name] = metric_func(row)
        metric_results.append(result_row)

# %% [markdown]
# ### Logging

# %%
# log data
import wandb
from dotenv import load_dotenv
load_dotenv()
wandb.login(relogin=True)

# %%
opentable_artifact = wandb.Artifact("opentable_trajectories_gptv.v0", type="dataset")
# opentable_table = wandb.Table(dataframe=traj_df_new)

opentable_table_eligible = wandb.Table(dataframe=eligible_traj_df)
#
# opentable_artifact.add(opentable_table, "opentable")
opentable_artifact.add(opentable_table_eligible, "opentable-eligibleonly")
# opentable_artifact.add_file(str(dataset_path/'csv/data+gptv.csv'))
opentable_artifact.add_file(str(dataset_path/'csv/data+gptv-eligible.csv'))

# Log the table to visualize with a run...
run.log({"opentable_gptv_generation": opentable_table_eligible})



