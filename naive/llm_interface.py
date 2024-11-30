from abc import ABC, abstractmethod
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional, Dict, Any

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt"""
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for multiple prompts"""
        pass

class GeminiInterface(LLMInterface):
    """Gemini API interface"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.safety_settings=[
             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
        ]
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def batch_generate(self, prompts: list[str]) -> list[str]:
        return [self.generate_response(prompt) for prompt in prompts]
