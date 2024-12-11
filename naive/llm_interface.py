from abc import ABC, abstractmethod
import google.generativeai as genai
from typing import Optional, Dict, Any, List
from openai import OpenAI


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt"""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        pass


class GeminiInterface(LLMInterface):
    """Gemini API interface"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate_response(prompt) for prompt in prompts]


class ChatGPTInterface(LLMInterface):
    """ChatGPT API interface"""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            # Explicitly check if message content exists
            message_content = response.choices[0].message.content
            if message_content is None:
                print(f"No content returned for prompt: {prompt}")
                return ""
            return message_content
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            responses.append(self.generate_response(prompt))
        return responses

class LlamaInterface(LLMInterface):
    """Llama 3 interface"""

    def __init__(self, api_key: str):
        from huggingface_hub import login
        login(api_key)

        self.model_name = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate_response(self, prompt: str) -> str:
        try:
            outputs = self.model.generate(inputs["input_ids"], max_length=200)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Explicitly check if message content exists
            message_content = response
            if message_content is None:
                print(f"No content returned for prompt: {prompt}")
                return ""
            return message_content
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            responses.append(self.generate_response(prompt))
        return responses