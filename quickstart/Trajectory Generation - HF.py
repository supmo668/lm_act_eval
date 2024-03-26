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
dataset_path = Path(f"lm_act_eval/.cache/{dataset_name}")
assert dataset_path.exists()
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
from lm_act_eval.evaluation_harness.evaluators.sft.utils import generate_text_and_merge, text_from_prompt_and_action

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
model_save_name = model_name.split('/')[1]
try:
  traj_df_new_gen.to_csv(dataset_path/f'csv/data+gen_{model_save_name}.csv', index=False)
except:
  traj_df_new_gen.to_csv(dataset_path/f'csv/data+eval_gen.csv', index=False)


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
run.log({"opentable_gptv_generation": eval_generation})


# %%
# Log the table to visualize with a run...
run.log({"eval-generations": eval_generation})
# and Log as an Artifact to increase the available row limit!
run.log_artifact(opentable_artifact)