from portkey_ai import Portkey
from portkey_ai import AsyncPortkey

import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def match_model_to_virtual_key_env(model):
  match model:
    case "gemini-1.5-pro-latest":
      key = os.getenv("GEMINI_VIRTUAL_KEY")
    case "gpt4":
      key = os.getenv("GPT4_VIRTUAL_KEY")
    case "gpt4-vision":
      key = os.getenv("GPT4V_VIRTUAL_KEY")
    case "gpt3.5":
      key = os.getenv("GPT3.5_VIRTUAL_KEY")
  assert key, f"Matching key not found for model {model}"
  return key

async def query_model(
    user_message, model='gemini-1.5-pro-latest'
    ):
    """
    Reads a message from a file and sends it to the Google Gemini model via Portkey.
    
    Args:
    file_path (str): Path to the text file containing the user's message.
    """
    # Initialize Portkey with your API and virtual keys from environment variables
    os.environ["PORTKEY_API_KEY"] = await match_model_to_virtual_key_env(model)
    portkey = AsyncPortkey(
        api_key=os.getenv("PORTKEY_API_KEY"),  # 
        virtual_key=os.getenv("VIRTUAL_KEY")   # 
    )
    if model.lower().startswith("gpt4"):
      portkey=portkey.with_options(
        config=os.getenv("PORTKEY_GPT4_CONFIG"))
    # Invoke chat completions with Google Gemini
    completion = await portkey.chat.completions.create(
        messages=[
            {"role": "user", "content": user_message}
        ],
        model=model
    )
    # Extract and print the response
    return completion.choices[0].message.content