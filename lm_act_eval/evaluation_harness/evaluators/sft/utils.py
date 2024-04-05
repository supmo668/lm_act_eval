from sqlalchemy import desc
from tqdm import tqdm
import warnings 

from transformers import pipeline, PreTrainedTokenizerBase

from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate


def custom_collate(batch):
    batch = [item for item in batch if item is not None]  # Filter out None values
    return default_collate(batch)

def text_from_prompt_and_action(
    prompt: str, action: str, tokenizer: PreTrainedTokenizerBase
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": action},
    ]
    text = tokenizer.apply_chat_template(  # type: ignore
        conversation=messages, tokenize=False
    )
    assert isinstance(text, str)
    return text


def generate_completions(tokenizer, model, prompts: list[str]) -> list[str]:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,  # type: ignore
        batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = pipe(
        [text_from_prompt_and_action(prompt, "", tokenizer) for prompt in prompts],
        return_full_text=False,
        max_new_tokens=256,
    )
    return [
        output[0]["generated_text"] for output in tqdm(out, desc="Generating for Evals")
    ]


def generate_text_and_merge(
    df,
    input_text_column,
    model,
    tokenizer,
    device,
    max_new_tokens=256,
    batch_size=8,
    **generation_kwargs
):
    """
    Generate text and merge it with the original DataFrame.

    Args:
        df (pandas.DataFrame): The original DataFrame.
        input_text_column (str): The name of the column containing the input text.
        model (torch.nn.Module): The model used for text generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text encoding.
        device (torch.device): The device on which the model and tensors are located.
        max_length (int, optional): The maximum length of the generated text. Defaults to 256.
        batch_size (int, optional): The batch size for processing the input text. Defaults to 8.
        **generation_kwargs: Additional keyword arguments to be passed to the model's generate method.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column containing the generated text.
    """
    df = df.copy().dropna(subset=[input_text_column])
    if len(df) != len(df):
        warnings.warn(f"{len(df) - len(df)} rows dropped due to NaN in {input_text_column}.")
    text_dataset = Dataset.from_pandas(df[[input_text_column]])
    text_dataloader = DataLoader(
        text_dataset, batch_size=batch_size, collate_fn=custom_collate)
    
    generated_texts = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc=f"Generate from {input_text_column}"):
            input_texts = batch[input_text_column]
            encoding = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,  # Ensure inputs do not exceed model's max length
            )
            generation_kwargs.setdefault('pad_token_id', tokenizer.eos_token_id)
            # Generate text outputs
            output_ids = model.generate(
                encoding.input_ids.to(device), 
                max_new_tokens=max_new_tokens, 
                attention_mask=encoding.attention_mask.to(device),
                **generation_kwargs
            )
            for ids in output_ids:
                response_text = tokenizer.decode(
                    ids, skip_special_tokens=True)
                generated_texts.append(response_text)

    # Merge the generated text with the original DataFrame
    df["generated_"+input_text_column] = generated_texts
    return df


from lm_act_eval.evaluation_harness.helper_functions.utils import function_registry
from lm_act_eval.evaluation_harness.evaluators.registry import evaluator_registry
from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry

from typing import *
from .base import BaseEvaluator
def cfg_to_function(funct_pairs: OmegaConf | Dict[str, Dict[str, str]]) -> Generator[Tuple[Tuple[str, str], str], None, None]:
    """
    A function that processes a function query along with a function pair and returns a tuple containing a source field, target field, and function name.

    Parameters:
        function_query (Dict): A dictionary containing the function query.
        funct_pair (List[Dict]): A list of dictionaries containing function pairs.

    Returns:
        Tuple[Tuple[str, str], str]: A tuple containing a tuple with source field and target field, and a string representing the function name.
    """
    for tgt_field, function_query in funct_pairs.items():
        src_field, func_name = dict(function_query).popitem()
        # Immediately yield an identity function for None func_name and skip further processing
        if func_name is None:
            yield (src_field, tgt_field), lambda x: x
            continue  # Skip to the next iteration
        # Determine if a class method is specified and split accordingly
        _function_is_cls = '.' in func_name
        if _function_is_cls:
            func_name, cls_func = func_name.split(".")

        # Retrieve function or class method
        function = function_registry.get(func_name)
        if _function_is_cls:
            function = getattr(function(), cls_func)

        yield (src_field, tgt_field), function

    
def cfg_to_evaluator(eval_pairs: OmegaConf | Dict[str, Dict[str, str]]) -> Generator[Tuple[BaseEvaluator, list[str]], None, None]:
  for metric_name, eval_params in eval_pairs.items():
    eval_metric = metric_registry.get(metric_name)
    # yield the evaluator
    yield eval_metric(eval_params.args), eval_params.inputs