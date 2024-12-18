import sys
import os
import logging
import argparse
import pandas as pd
import numpy as np
import importlib.util
import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Dynamically import the module
module_path = os.path.join(project_root, "retrieve_with_predictions", "bm25_scratch.py")
spec = importlib.util.spec_from_file_location("bm25_scratch", module_path)
bm25_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm25_module)
MODEL_DIR = os.path.join(project_root, "models")
from src.model import LinkPredictor, load_model

# Use the imported module
BM25MultiHopRetriever = bm25_module.BM25MultiHopRetriever
from src.dataset import (
    get_condensed_frames_dataset,
    get_random_question,
    get_frames_relevant_dataset,
)

# Optional LLM imports with graceful degradation
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

# Sentence Transformers for answer similarity
from sentence_transformers import SentenceTransformer, util


class LLMFactory:
    """Factory for creating and managing different LLM instances."""

    @staticmethod
    def create_llm(model_name, api_key=None):
        """
        Create an LLM instance based on the model name.

        Args:
            model_name (str): Name of the LLM model
            api_key (str, optional): API key for the LLM

        Returns:
            An LLM model instance or None if not supported
        """
        try:
            if model_name.startswith("gemini"):
                if not genai:
                    print(
                        "Warning: Google Generative AI not installed. Skipping Gemini."
                    )
                    return None

                # Configure Gemini
                if not api_key:
                    api_key = os.getenv("GEMINI_API_KEY")

                if not api_key:
                    print("Warning: No Gemini API key found. Skipping Gemini.")
                    return None

                genai.configure(api_key=api_key)
                return genai.GenerativeModel(model_name)

            elif model_name.startswith("gpt"):
                if not openai:
                    print("Warning: OpenAI library not installed. Skipping GPT.")
                    return None

                # Configure OpenAI
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")

                if not api_key:
                    print("Warning: No OpenAI API key found. Skipping GPT.")
                    return None

                openai.api_key = api_key
                return openai.ChatCompletion

            else:
                print(f"Warning: Unsupported LLM model: {model_name}")
                return None

        except Exception as e:
            print(f"Error initializing LLM {model_name}: {e}")
            return None


class BaseRetrieverTester:
    """
    A comprehensive testing framework for Base Retriever with multiple LLMs.
    """

    def __init__(self, dataset=None, log_level=logging.INFO):
        """
        Initialize the tester with optional dataset.

        Args:
            dataset (pd.DataFrame, optional): Dataset to test on.
                                              If None, loads default dataset.
            log_level (int): Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load dataset
        self.dataset = (
            dataset if dataset is not None else get_condensed_frames_dataset()
            # dataset if dataset is not None else get_frames_relevant_dataset()
            # dataset if dataset is not None else get_random_question()
        )

        # Initialize sentence transformer for similarity
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.hop_model, self.vectorizer = load_model(MODEL_DIR, input_size=300)

        # Initialize metrics storage
        self.results = {
            "model": [],
            "prompt": [],
            "retrieved_docs": [],
            "ground_truth_answer": [],
            "generated_answer": [],
            "answer_similarity": [],
            "answer_accuracy": [],
            "retrieval_iterations": [],
            "total_tokens_used": [],
            "query_count": [],
        }

    def predict_hop_count(self, prompt):
        """Predict the number of hops required for retrieval"""
        try:
            # Vectorize the input
            features = self.vectorizer.transform([prompt]).toarray()

            # Convert to tensor and predict
            input_tensor = torch.tensor(features, dtype=torch.float32)
            hop_count = self.hop_model(input_tensor).item()

            # Round to nearest integer and ensure it's at least 1
            return max(1, round(hop_count))
        except Exception as e:
            self.logger.error(f"Error predicting hop count: {e}")
            return 1  # Fallback to 1 hop

    def calculate_answer_similarity(
        self, ground_truth, generated_answer, threshold=0.8
    ):
        """
        Calculate semantic similarity between ground truth and generated answer.

        Args:
            ground_truth (str): Original answer from dataset
            generated_answer (str): Answer generated by retriever
            threshold (float): Similarity threshold for accuracy

        Returns:
            Tuple of (similarity_score, is_accurate)
        """
        # Encode sentences
        ground_truth_embedding = self.similarity_model.encode(ground_truth)
        generated_answer_embedding = self.similarity_model.encode(generated_answer)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(
            ground_truth_embedding, generated_answer_embedding
        ).item()

        # Determine accuracy based on threshold
        is_accurate = similarity >= threshold

        return similarity, is_accurate

    def test_retriever(
        self,
        model_names=["gemini-pro"],
        num_samples=None,
        docs_per_query=1,
        similarity_threshold=0.8,
    ):
        """
        Test retriever with different LLMs.

        Args:
            model_names (list): List of LLM model names to test
            num_samples (int, optional): Number of samples to test.
                                         If None, test all samples.
            num_iterations (int): Number of retrieval iterations
            docs_per_query (int): Maximum documents to retrieve per query
            similarity_threshold (float): Threshold for answer accuracy
        """
        # Limit samples if specified
        test_data = self.dataset.sample(n=num_samples) if num_samples else self.dataset

        for model_name in model_names:
            try:
                # Create LLM instance
                llm = LLMFactory.create_llm(model_name)

                # Skip if LLM not supported or configured
                if llm is None:
                    self.logger.warning(
                        f"Skipping model {model_name} due to configuration issues."
                    )
                    continue

                # Create BM25 retriever with this LLM
                retriever = BM25MultiHopRetriever(llm_provider="gemini")

                # Test each prompt
                for idx, row in test_data.iterrows():
                    prompt = row["Prompt"]
                    ground_truth_answer = row["Answer"]
                    query_count = row["query_count"]
                    predicted_hops = self.predict_hop_count(prompt)
                    self.logger.info(f"Predicted hops: {predicted_hops}")
                    # Increment predicted hops
                    predicted_hops = predicted_hops + 1
                    # Perform retrieval
                    answer, retrieved_docs, _, _, _, _ = retriever.retrieve(
                        prompt,
                        ground_truth_answer=ground_truth_answer,
                        num_iterations=predicted_hops,
                        docs_per_query=docs_per_query,
                    )
                    self.logger.info(f"Generated answer: {answer}")

                    # Calculate answer similarity
                    similarity, is_accurate = self.calculate_answer_similarity(
                        ground_truth_answer, answer, threshold=similarity_threshold
                    )

                    # Store results
                    self.results["model"].append(model_name)
                    self.results["prompt"].append(prompt)
                    self.results["retrieved_docs"].append(retrieved_docs)
                    self.results["ground_truth_answer"].append(ground_truth_answer)
                    self.results["generated_answer"].append(answer)
                    self.results["answer_similarity"].append(similarity)
                    self.results["answer_accuracy"].append(is_accurate)
                    self.results["retrieval_iterations"].append(len(retrieved_docs))
                    self.results["total_tokens_used"].append(None)
                    self.results["query_count"].append(query_count)

            except Exception as e:
                self.logger.error(f"Error testing {model_name}: {e}")

        return pd.DataFrame(self.results)

    def generate_report(self, results_df):
        """
        Generate a comprehensive report from test results.

        Args:
            results_df (pd.DataFrame): Results dataframe from test_retriever

        Returns:
            dict: Summary statistics
        """
        report = {}

        # Aggregate metrics per model
        for model in results_df["model"].unique():
            model_results = results_df[results_df["model"] == model]

            report[model] = {
                "total_queries": len(model_results),
                "avg_retrieved_docs": model_results["retrieved_docs"].apply(len).mean(),
                "avg_retrieval_iterations": model_results[
                    "retrieval_iterations"
                ].mean(),
                "avg_answer_similarity": model_results["answer_similarity"].mean(),
                "accuracy_rate": model_results["answer_accuracy"].mean(),
            }

        return report

    def generate_batch_report(self, results_df):
        report = {}
        for query_count in results_df["query_count"].unique():
            model_results = results_df[results_df["query_count"] == query_count]

            report[query_count] = {
                "total_queries": len(model_results),
                "avg_retrieved_docs": model_results["retrieved_docs"].apply(len).mean(),
                "avg_retrieval_iterations": model_results[
                    "retrieval_iterations"
                ].mean(),
                "avg_answer_similarity": model_results["answer_similarity"].mean(),
                "accuracy_rate": model_results["answer_accuracy"].mean(),
            }

        return report


def main():
    parser = argparse.ArgumentParser(description="Base Retriever Multi-LLM Tester")
    parser.add_argument(
        "--models", nargs="+", default=["gemini-pro"], help="LLM models to test"
    )
    parser.add_argument(
        "--samples", type=int, default=None, help="Number of samples to test"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of retrieval iterations(hops)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for answer accuracy",
    )

    args = parser.parse_args()

    # Initialize and run tester
    tester = BaseRetrieverTester()
    results = tester.test_retriever(
        model_names=args.models,
        num_samples=args.samples,
        similarity_threshold=args.similarity_threshold,
    )

    # Generate and print report
    report = tester.generate_report(results)
    print("\n--- Base Retriever Multi-LLM Test Report ---")
    for model, metrics in report.items():
        print(f"\nModel: {model}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    batch_report = tester.generate_batch_report(results)
    print("\n--- Per Batch Report ---")
    for batch, metrics in batch_report.items():
        print(f"\nBatch: {batch}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    # Optional: Save results to CSV
    # results.to_csv("base_retriever_test_results_5.csv", index=False)


if __name__ == "__main__":
    main()
