import os
from llm_interface import ChatGPTInterface

def main():
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Initialize ChatGPT interface
    chatgpt = ChatGPTInterface(api_key)

    # Test question
    question = "What is the capital of France and what is its most famous landmark?"

    # Get response
    print("Question:", question)
    print("\nGenerating response...\n")
    
    response = chatgpt.generate_response(question)
    print("Response:", response)

if __name__ == "__main__":
    main()
