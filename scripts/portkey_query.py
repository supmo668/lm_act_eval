from portkey_ai import Portkey
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Portkey with your API and virtual keys from environment variables
portkey = Portkey(
    api_key=os.getenv("PORTKEY_API_KEY"),  # Load Portkey API key from environment variables
    virtual_key=os.getenv("VIRTUAL_KEY")   # Load virtual key for Google from environment variables
)

def query_google_gemini_from_file(file_path):
    """
    Reads a message from a file and sends it to the Google Gemini model via Portkey.
    
    Args:
    file_path (str): Path to the text file containing the user's message.
    """
    try:
        # Read the user's message from the file
        with open(file_path, 'r') as file:
            user_message = file.read().strip()
        
        # Invoke chat completions with Google Gemini
        completion = portkey.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            model='gemini-pro'
        )
        
        # Extract and print the response
        print("Response from Google Gemini:")
        for message in completion['messages']:
            if message['role'] == 'ai':
                print(message['content'])
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
  from pathlib import Path
  query_google_gemini_from_file(Path(__file__).parent/"asset/query.txt")
