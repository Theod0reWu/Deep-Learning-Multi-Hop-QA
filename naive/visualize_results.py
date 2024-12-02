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
    
    # Get common reasoning types between both models
    gemini_types = set(gemini_data.keys())
    gpt_types = set(gpt_data.keys())
    common_types = gemini_types.intersection(gpt_types)
    
    # Process Gemini data for common types only
    for reasoning_type in common_types:
        values = gemini_data[reasoning_type]
        if isinstance(values, dict) and "accuracy" in values:
            metrics.append(
                {
                    "model": "Gemini",
                    "reasoning_type": reasoning_type,
                    "accuracy": values["accuracy"],
                    "mean_similarity": values["mean_similarity"],
                }
            )
    
    # Process GPT data for common types only
    for reasoning_type in common_types:
        values = gpt_data[reasoning_type]
        if isinstance(values, dict) and "accuracy" in values:
            metrics.append(
                {
                    "model": "GPT",
                    "reasoning_type": reasoning_type,
                    "accuracy": values["accuracy"],
                    "mean_similarity": values["mean_similarity"],
                }
            )
    
    # Print statistics about reasoning types
    print("\nReasoning Type Statistics:")
    print(f"Gemini unique types: {len(gemini_types - gpt_types)}")
    print(f"GPT unique types: {len(gpt_types - gemini_types)}")
    print(f"Common types: {len(common_types)}")
    
    return pd.DataFrame(metrics)


def create_visualizations(df):
    # Set the style
    plt.style.use("default")
    sns.set_theme(style="whitegrid")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create plots for each model
    models = df['model'].unique()
    
    for idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        
        # Get top 5 reasoning types for this model
        top_types = model_df.groupby("reasoning_type")["accuracy"].mean().nlargest(5)
        
        # Create histogram for top 5 accuracies
        sns.barplot(
            data=model_df[model_df['reasoning_type'].isin(top_types.index)],
            x="reasoning_type",
            y="accuracy",
            ax=axes[idx]
        )
        
        axes[idx].set_title(f"Top 5 Reasoning Types Accuracy for {model}", pad=20)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")
        axes[idx].set_ylim(0, 1)
        axes[idx].set_ylabel("Accuracy")
        axes[idx].set_xlabel("Reasoning Type")

    plt.suptitle("Model Performance Comparison: Top 5 Accuracies", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig("model_top5_accuracies.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_performance(df):
    # Calculate overall statistics
    overall_stats = df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'mean_similarity': ['mean', 'std']
    }).round(4)
    
    print("\nOverall Performance Statistics:")
    print(overall_stats)
    
    # Find best performing reasoning types for each model
    best_accuracy = df.sort_values('accuracy', ascending=False).groupby('model').head(3)
    print("\nTop 3 Best Performing Reasoning Types by Accuracy:")
    for model in ['Gemini', 'GPT']:
        print(f"\n{model}:")
        model_best = best_accuracy[best_accuracy['model'] == model]
        for _, row in model_best.iterrows():
            print(f"- {row['reasoning_type']}: {row['accuracy']:.4f} accuracy, {row['mean_similarity']:.4f} similarity")
    
    # Calculate performance gap
    avg_by_type = df.pivot(index='reasoning_type', columns='model', values=['accuracy', 'mean_similarity'])
    gaps = (avg_by_type['accuracy']['Gemini'] - avg_by_type['accuracy']['GPT']).sort_values(ascending=False)
    
    print("\nBiggest Performance Gaps (Gemini - GPT):")
    print("\nGemini Advantage:")
    for type_, gap in gaps.head(3).items():
        print(f"- {type_}: {gap:.4f}")
    print("\nGPT Advantage:")
    for type_, gap in gaps.tail(3).items():
        print(f"- {type_}: {gap:.4f}")


def main():
    # Load results
    gemini_results = load_results("results/gemini_results.txt")
    gpt_results = load_results("results/gpt_results.txt")
    
    # Prepare data
    df = prepare_data(gemini_results, gpt_results)
    
    # Create visualizations
    create_visualizations(df)
    print("Visualization has been saved as 'model_top5_accuracies.png'")
    
    # Analyze performance
    analyze_performance(df)

    # Print some statistics
    print("\nNumber of data points:")
    print(df.groupby("model").size())


if __name__ == "__main__":
    main()
