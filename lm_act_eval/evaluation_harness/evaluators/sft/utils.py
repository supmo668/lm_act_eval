from sqlalchemy import desc
from tqdm import tqdm

from transformers import pipeline, PreTrainedTokenizerBase

from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
import torch


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
    text_dataset = Dataset.from_pandas(pd.DataFrame(df[input_text_column]))
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)

    generated_texts = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc=f"Generate from {input_text_column}"):
            input_texts = batch[input_text_column]
            input_ids = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            # Generate text outputs
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, **generation_kwargs
            )

            for ids in output_ids:
                response_text = tokenizer.decode(
                    ids, skip_special_tokens=True)
                generated_texts.append(response_text)

    # Merge the generated text with the original DataFrame
    df["generated_"+input_text_column] = generated_texts
    return df
