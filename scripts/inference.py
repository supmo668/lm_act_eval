import click
from dotenv import load_dotenv
import pandas as pd

import wandb

from lm_act_eval.common.utils import dataframe_to_dataset
from lm_act_eval.common.hf import load_model_and_tokenizer
from lm_act_eval.evaluation_harness.evaluators.sft.utils import generate_text_and_merge
from lm_act_eval.evaluation_harness.constants import CACHE_DIR, DATETIME_STR

# multi-gpu
from accelerate import Accelerator
accelerator = Accelerator()

load_dotenv()  # Load environment variables from .env file if present

@click.command()
@click.option(
  '--model-name', default="MULTION-AI/feasible-morning-17-v4", help="The model name or path.")
@click.option(
  '--csv-path', prompt=True, 
  default='.cache/five-star-trajectories/csv/data+gptv.csv', 
  help="Path to the CSV file.")
@click.option(
  '--input-text-column', prompt=True,
  default='chat_completion_messages_content',
  help="The name of the column containing input text.")
@click.option('--max-length', default=256, help="Maximum length of the generated text.")
def main(model_name, csv_path, input_text_column, max_length):
    run_name = f"lm-act-eval_inference_model:{model_name}_{DATETIME_STR}"
    # wandb.init(
    #   project="trajectory_eval", entity="multion-agi", name=run_name)
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    dataset, df = dataframe_to_dataset(csv_path)
    df_merged = generate_text_and_merge(dataset, df, input_text_column, model, tokenizer, device, max_length, batch_size=2)
    
    # log
        # Log the entire DataFrame to wandb
    # wandb.log({
    #   "final_dataframe": wandb.Table(dataframe=df_merged)})
    
    # save the merged DataFrame to a new CSV file
    df_merged.to_csv(
      CACHE_DIR/f"{run_name}.csv", index=False)
      
if __name__ == "__main__":
    main()

wandb.finish()

