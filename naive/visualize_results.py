import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import re


def parse_json_metrics(content):
    # Find the main JSON object that contains the metrics
    start = content.find("{", content.find("Metrics by Reasoning Type:"))
    if start == -1:
        return {}

    # Count braces to find the matching closing brace
    count = 1
    end = start + 1
    while count > 0 and end < len(content):
        if content[end] == "{":
            count += 1
        elif content[end] == "}":
            count -= 1
        end += 1

    if count > 0:
        return {}

    try:
        json_str = content[start:end]
        data = json.loads(json_str)
        metrics = {}
        for key, value in data.items():
            if (
                isinstance(value, dict)
                and "accuracy" in value
                and "mean_similarity" in value
            ):
                metrics[key] = {
                    "accuracy": value["accuracy"],
                    "mean_similarity": value["mean_similarity"],
                }
        return metrics
    except json.JSONDecodeError:
        print("Failed to parse JSON from Gemini results")
        return {}


def parse_text_metrics(content):
    metrics = {}
    current_type = None

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("===") or line.endswith("==="):
            continue

        if ":" in line:
            if line.startswith("accuracy:") or line.startswith("mean_similarity:"):
                key, value = line.split(":")
                key = key.strip()
                value = float(value.strip())

                if current_type is None:
                    continue  # Skip overall metrics
                else:
                    if current_type not in metrics:
                        metrics[current_type] = {}
                    metrics[current_type][key] = value
            elif not any(
                metric in line.lower() for metric in ["accuracy:", "mean_similarity:"]
            ):
                current_type = line.split(":")[0].strip()

    return metrics


def load_results(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Try JSON parsing first
    metrics = parse_json_metrics(content)
    if not metrics:
        # If JSON parsing fails, try text parsing
        metrics = parse_text_metrics(content)

    return metrics


def prepare_data(gemini_data, gpt_data):
    # Extract reasoning types and metrics
    metrics = []

    # Process Gemini data
    for reasoning_type, values in gemini_data.items():
        if isinstance(values, dict) and "accuracy" in values:
            metrics.append(
                {
                    "model": "Gemini",
                    "reasoning_type": reasoning_type,
                    "accuracy": values["accuracy"],
                    "mean_similarity": values["mean_similarity"],
                }
            )

    # Process GPT data
    for reasoning_type, values in gpt_data.items():
        if isinstance(values, dict) and "accuracy" in values:
            metrics.append(
                {
                    "model": "GPT",
                    "reasoning_type": reasoning_type,
                    "accuracy": values["accuracy"],
                    "mean_similarity": values["mean_similarity"],
                }
            )

    return pd.DataFrame(metrics)


def create_visualizations(df):
    # Set the style
    plt.style.use("default")
    sns.set_theme(style="whitegrid")

    # Set up the figure size
    plt.figure(figsize=(15, 10))

    # 1. Overall Performance Comparison
    top_types = df.groupby("reasoning_type")["accuracy"].mean().nlargest(5).index
    top_df = df[df["reasoning_type"].isin(top_types)]

    plt.subplot(2, 1, 1)
    sns.barplot(data=top_df, x="reasoning_type", y="accuracy", hue="model")
    plt.title("Top 5 Reasoning Types: Accuracy Comparison", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="Model")

    # 2. Accuracy vs Mean Similarity Scatter Plot
    plt.subplot(2, 1, 2)
    sns.scatterplot(
        data=df, x="accuracy", y="mean_similarity", hue="model", style="model", s=100
    )
    plt.title("Accuracy vs Mean Similarity by Model", pad=20)
    plt.xlabel("Accuracy")
    plt.ylabel("Mean Similarity")
    plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Load results
    gemini_results = load_results("results/gemini_results.txt")
    gpt_results = load_results("results/gpt_results.txt")  # Use the new format

    # Prepare data
    df = prepare_data(gemini_results, gpt_results)

    # Create visualizations
    create_visualizations(df)
    print("Visualizations have been saved as 'model_comparison.png'")

    # Print some statistics
    print("\nNumber of data points:")
    print(df.groupby("model").size())


if __name__ == "__main__":
    main()
