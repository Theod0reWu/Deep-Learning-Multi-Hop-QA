import os
from llm_interface import GeminiInterface

def main():
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")

    # Initialize Gemini interface
    gemini = GeminiInterface(api_key)

    # Test question
    question = "What is the capital of France and what is its most famous landmark?"

    # Get response
    print("Question:", question)
    print("\nGenerating response...\n")
    
    response = gemini.generate_response(question)
    print("Response:", response)

if __name__ == "__main__":
    main()
