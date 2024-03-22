from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import torch

import warnings
from accelerate import Accelerator

accelerator = Accelerator()

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name:str ):
    """
    Load the model and tokenizer for the given model HF name.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        Tuple: A tuple containing the loaded model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", load_in_8bit=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count())))  # Wrap the model for parallel processing
    return model, tokenizer, device

# Function to perform the text generation
def generate_text(input_text, model, tokenizer, max_new_tokens=256, log=True):
    """
    Generate text based on input using the provided model and tokenizer.
    
    Parameters:
    input_text (str): The input text to generate text from.
    model (torch.nn.Module): The model used for text generation.
    tokenizer: The tokenizer used to process the input text.
    device: The device the model is being run on.
    max_length (int): The maximum length of the generated text (default is 256).
    
    Returns:
    str: The generated text based on the input.
    """
    if wandb.run is None and log:
        warnings.warn("Wandb is not initialized. Generated text will not be logged.")

    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(
            input_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids, max_length=max_new_tokens)[0]
        response_text = tokenizer.decode(
            output_ids, skip_special_tokens=True)
        if log and wandb.run is not None:
            wandb.log({"Generated Text": response_text})  # Log the generated text to wandb
        return response_text