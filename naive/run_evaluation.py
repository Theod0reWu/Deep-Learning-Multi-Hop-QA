import sys
import os
import pandas as pd
import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_frames_dataset
from llm_interface import GeminiInterface
from evaluator import Evaluator
from sklearn.model_selection import train_test_split


def main():
    # Load API key (you should set this in your environment)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")

    # Load dataset
    print("Loading dataset...")
    df = get_frames_dataset()

    # Ensure we have the required columns
    required_cols = ["Prompt", "Answer", "reasoning_types"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset missing required columns: {required_cols}")

    # Split dataset (optional, remove if not needed)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Ensure we have DataFrames
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    # Initialize model
    model = GeminiInterface(api_key=api_key)

    # Initialize evaluator
    evaluator = Evaluator(model)

    # Run evaluation
    results = evaluator.evaluate(test_df)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.txt")

    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")

        f.write("Overall Metrics:\n")
        f.write("--------------\n")
        f.write(json.dumps(results["overall"], indent=2))
        f.write("\n\n")

        f.write("Metrics by Reasoning Type:\n")
        f.write("------------------------\n")
        f.write(json.dumps(results["by_type"], indent=2))

    print(f"\nResults saved to: {results_file}")

    # Print results to console as well
    print("\nOverall Metrics:")
    print(json.dumps(results["overall"], indent=2))

    print("\nMetrics by Reasoning Type:")
    print(json.dumps(results["by_type"], indent=2))


if __name__ == "__main__":
    main()
