from portkey_ai import Portkey
from portkey_ai import AsyncPortkey

import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()


# Initialize Portkey with your API and virtual keys from environment variables
portkey = AsyncPortkey(
    api_key=os.getenv("GEMINI_VIRTUAL_KEY"),  # Load Portkey API key from environment variables
    virtual_key=os.getenv("VIRTUAL_KEY")   # Load virtual key for Google from environment variables
)

async def query_google_gemini_from_file(
    file_path, model='gemini-1.5-pro-latest'
    ):
    """
    Reads a message from a file and sends it to the Google Gemini model via Portkey.
    
    Args:
    file_path (str): Path to the text file containing the user's message.
    """
    # Read the user's message from the file
    with open(file_path, 'r') as file:
        user_message = file.read().strip()
    
    # Invoke chat completions with Google Gemini
    completion = await portkey.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a smart & experienced software development assistant."},
            {"role": "user", "content": user_message}
        ],
        model=model
    )
    
    # Extract and print the response
    print(f"Response from Google Gemini written to: {file_path.parent/f'{file_path.stem}.resp.md'})")
    message = completion.choices[0].message.content
    with open(file_path.parent/f"{file_path.stem}.resp.md", 'w') as file:
        file.write(message)
# Example usage
if __name__ == "__main__":
  from pathlib import Path
  asyncio.run(query_google_gemini_from_file(Path(__file__).parent/"asset/query.txt"))

  
