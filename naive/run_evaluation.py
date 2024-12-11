import sys
import os
import pandas as pd
import datetime
import json
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_frames_dataset
from llm_interface import GeminiInterface, ChatGPTInterface, LlamaInterface
from evaluator import Evaluator


def get_model(model_name: str):
    """Initialize and return the specified model"""
    if model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        return GeminiInterface(api_key), "Gemini"
    elif model_name == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        return ChatGPTInterface(api_key), "ChatGPT"
    elif model_name == 'llama':
        api_key = os.getenv("HUGGING_FACE_API_KEY")
        if not api_key:
            raise ValueError("Please set HUGGING_FACE_API_KEY environment variable")
        return LlamaInterface(api_key), "Llama"
    else:
        raise ValueError("Invalid model name. Choose 'gemini' or 'gpt'")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run evaluation with specified LLM")
    parser.add_argument(
        "model",
        type=str,
        choices=["gemini", "gpt", "llama"],
        help="Which model to use (gemini, gpt or llama)",
    )
    args = parser.parse_args()

    print("Initializing the model")
    # Initialize the specified model
    model, model_name = get_model(args.model)

    print("loading the dataset")
    # Load test dataset
    test_df = get_frames_dataset()

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Get current timestamp for the results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        results_dir, f"evaluation_results_{model_name}_{timestamp}.txt"
    )

    # Evaluate model
    evaluator = Evaluator(model, model_name)
    results = evaluator.evaluate(test_df)

    print(results)

    # Save results
    with open(results_file, "w") as f:
        f.write(f"=== {model_name} Evaluation Results ===\n\n")

        f.write("Overall Metrics:\n")
        for metric, value in results["overall"].items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")

        f.write("\nMetrics by Reasoning Type:\n")
        for rtype, metrics in results["by_type"].items():
            f.write(f"\n{rtype}:\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
