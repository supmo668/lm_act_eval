import tqdm

from transformers import pipeline, PreTrainedTokenizerBase


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