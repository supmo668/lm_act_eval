import base64
import requests
import re
import os
import io
from urllib.parse import urlparse
import mimetypes

from openai import OpenAI
from PIL import Image

from finetuning.finetuning.evaluate.evaluators.vision.base import Pipeline
from finetuning.finetuning.evaluate.evaluators.vision.constants import ENDPOINTS

from pathlib import Path

from finetuning.finetuning.evaluate.evaluators.vision.config import gptv_config

class GPTV(Pipeline):
    def __init__(self, config: dict | gptv_config):
        try:
            self.api_key = os.environ["OPENAI_API_KEY"]
            self.openai_client = OpenAI()
        except KeyError:
            raise KeyError(
                "Please set your OpenAI API key in the environment variable `OPENAI_API_KEY`"
            )
        self.config = config
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        print(self.config)
        self.api_url = ENDPOINTS.get("Chat_Completions_API", "https://api.openai.com/v1/chat/completions")

    @staticmethod
    def is_base64(s):
        try:
            return base64.b64encode(base64.b64decode(s)).decode("utf-8") == s
        except Exception:
            return False

    @staticmethod
    def is_url(s):
        # Ensure the input is a string
        if not isinstance(s, str):
            return False
        try:
            result = urlparse(s)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def process_chat(self, image_path: str):
        return super().process_chat(image_path)
    
    def encode_image(self, image_input):
        if isinstance(image_input, str) and Path(image_input).exists():
            return image_input
        elif isinstance(image_input, Image.Image):
            buffered = io.BytesIO()
            image_input.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)} or check {image_input} exist")

    def process_image(self, text, image_input):
        if not isinstance(image_input, list):
            images = [image_input]
            
        image_contents = []
        for image in images:
            if self.is_url(image):
                image_content = {
                    "type": "image_url", "image_url": {
                        "url": image},
                }
            else:
                encoded_image = self.encode_image(image)
                mime_type, _ = mimetypes.guess_type(image[:30]) if isinstance(image, str) else ("image/jpeg", None)
                image_content = {
                    "type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
            image_contents.append(image_content)
        return image_contents
    
    def generate_completion(self, text, images, openai_sdk=True):
        """
        generate caption via OpenAI's SDK or API
        """
        image_content = self.process_image(
            text, images)
        composed_message = [{
            "role": "user", 
            "content": image_content}]
        if openai_sdk:
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=composed_message,
                max_tokens=self.config.max_tokens
            )
        else:
            payload = {
                "model": self.config.model,
                "messages": composed_message,
                "max_tokens": self.config.max_tokens
            }
            response = requests.post(
                self.api_url, headers=self.headers, json=payload)
            try:
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                print(f"Request failed: {e}")
        # may further extract meessage with: 
        # response.choices[0].message.content
        return response.choices
                
if __name__ == "__main__":
    """
    Check OpenAI docs on GPT-Vision: https://platform.openai.com/docs/guides/vision
    Standalone run:
        python -m finetuning.evaluate.vision.openai
    """
    from dotenv import load_dotenv
    assert load_dotenv(), "Please set your OpenAI API key in .env"
    image_src = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    ][0]
    prompt = "Describe the element and a detailed summary of what's in this webpage?"
    def run_example(pipeline, prompt, image_src):
        caption = pipeline.generate_completion(
            prompt, image_src)
        return caption
    pipeline = GPTV(gptv_config)
    print(run_example(
        pipeline, prompt, image_src))
    
    # [Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='This image depicts a serene natural landscape featuring a wooden boardwalk that extends through a lush meadow with green grasses. The vibrant greenery suggests this area may be a wetland or marsh, where such walkways are common to allow for exploration without disturbing the delicate ecosystem. The sky is blue with a scattering of white clouds, indicating it could be a fair-weather day and a wonderful opportunity for a nature walk. The photo gives a sense of tranquility and an invitation to the viewer to take a peaceful stroll amidst nature.', role='assistant', function_call=None, tool_calls=None))]
    #  extract meessage with: 
    #   response.choices[0].message.content