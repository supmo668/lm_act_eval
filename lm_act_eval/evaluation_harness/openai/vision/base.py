from abc import ABC, abstractmethod
from pathlib import Path

class Pipeline(ABC):
    @abstractmethod
    def process_image(self, image_path: str | Path):
        pass

    @abstractmethod
    def process_chat(self, text: str):
        return text
    
    @abstractmethod
    def generate_completion(self, text: str, image_path: str):
        pass

    def handle_output(self, result):
        return result