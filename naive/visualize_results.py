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
    metrics = {"combined": {}, "individual": {}}
    current_category = None
    
    print("Starting to parse metrics...")
    
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or line.startswith("===") or line.endswith("==="):
            continue
            
        # Skip overall metrics section
        if line.startswith("Overall Metrics:") or line.startswith("accuracy:") or line.startswith("mean_similarity:"):
            continue
            
        # Skip section headers
        if line == "Metrics by Reasoning Type:":
            continue
        
        # Detect category
        if line in ["combined:", "individual:"]:
            current_category = line[:-1]  # Remove the colon
            print(f"Found category: {current_category}")
            continue
        
        # Parse reasoning type and metrics
        if current_category and line.startswith("  "):  # This is a reasoning type line
            try:
                # Split on first colon to separate reasoning type from metrics
                reasoning_type, metrics_str = line.strip().split(": ", 1)
                
                # Clean up the metrics string and evaluate it
                metrics_str = metrics_str.replace("'", '"')  # Replace single quotes with double quotes
                metrics_dict = eval(metrics_str)  # Safely evaluate the dictionary string
                
                metrics[current_category][reasoning_type] = metrics_dict
                print(f"Parsed metrics for {current_category} - {reasoning_type}")
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error details: {str(e)}")
                continue
    
    print(f"Finished parsing. Found {len(metrics['combined'])} combined and {len(metrics['individual'])} individual metrics")
    return metrics


def load_results(filename):
    print(f"\nLoading results from {filename}")
    try:
        with open(filename, "r") as f:
            content = f.read()
        metrics = parse_text_metrics(content)
        print(f"Successfully loaded metrics from {filename}")
        return metrics
    except Exception as e:
        print(f"Error loading results from {filename}: {str(e)}")
        return {"combined": {}, "individual": {}}


def prepare_data(gemini_data, gpt_data):
    print("\nPreparing data from parsed results...")
    metrics = []
    
    for model_name, data in [("Gemini", gemini_data), ("GPT", gpt_data)]:
        print(f"\nProcessing {model_name} data:")
        for category in ["combined", "individual"]:
            if category in data:
                print(f"  Found {len(data[category])} entries in {category} category")
                for reasoning_type, values in data[category].items():
                    if isinstance(values, dict) and "accuracy" in values:
                        metrics.append({
                            "model": model_name,
                            "category": category,
                            "reasoning_type": reasoning_type,
                            "accuracy": values["accuracy"],
                            "mean_similarity": values["mean_similarity"]
                        })
                        print(f"    Added metrics for {reasoning_type}")
    
    df = pd.DataFrame(metrics)
    print(f"\nCreated DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
    return df


def create_visualizations(df):
    plt.style.use("default")
    sns.set_theme(style="whitegrid")

    # Create visualizations for both accuracy and mean_similarity
    metrics = ["accuracy", "mean_similarity"]
    
    # Create visualizations for both combined and individual categories
    for category in ["combined", "individual"]:
        category_df = df[df["category"] == category]
        
        for metric in metrics:
            # Create a figure with two subplots side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Create plots for each model
            models = category_df["model"].unique()

            for idx, model in enumerate(models):
                model_df = category_df[category_df["model"] == model]

                # Get top 5 reasoning types for this model based on the current metric
                top_types = model_df.groupby("reasoning_type")[metric].mean().nlargest(5)

                # Create histogram for top 5 values
                sns.barplot(
                    data=model_df[model_df["reasoning_type"].isin(top_types.index)],
                    x="reasoning_type",
                    y=metric,
                    ax=axes[idx],
                    color=["#2ecc71" if model == "Gemini" else "#3498db"][0]
                )

                axes[idx].set_title(f"Top 5 {category.title()} Types for {model}\n({metric.replace('_', ' ').title()})", pad=20)
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")
                axes[idx].set_ylim(0, 1)
                axes[idx].set_ylabel(metric.replace("_", " ").title())
                axes[idx].set_xlabel("Reasoning Type")

                # Add value labels on top of each bar
                for p in axes[idx].patches:
                    axes[idx].annotate(f'{p.get_height():.3f}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom')

            plt.suptitle(f"{category.title()} Types Performance: {metric.replace('_', ' ').title()}", y=1.05, fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{category}_{metric}_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()


def analyze_performance(df):
    # Calculate statistics for each category
    for category in ["combined", "individual"]:
        category_df = df[df["category"] == category]
        
        print(f"\n{category.upper()} CATEGORY STATISTICS:")
        print("=" * 50)
        
        # Overall statistics by model
        overall_stats = (
            category_df.groupby("model")
            .agg({"accuracy": ["mean", "std"], "mean_similarity": ["mean", "std"]})
            .round(4)
        )

        print("\nOverall Performance Statistics:")
        print(overall_stats)

        # Best performing reasoning types for each model
        print(f"\nTop 3 Best Performing {category.title()} Reasoning Types:")
        for model in ["Gemini", "GPT"]:
            print(f"\n{model}:")
            for metric in ["accuracy", "mean_similarity"]:
                best = (
                    category_df[category_df["model"] == model]
                    .sort_values(metric, ascending=False)
                    .head(3)
                )
                print(f"\nBest by {metric.replace('_', ' ')}:")
                for _, row in best.iterrows():
                    print(
                        f"- {row['reasoning_type']}: {row[metric]:.4f}"
                    )

        # Calculate performance gaps
        print(f"\nPerformance Gaps in {category.title()} Category (Gemini - GPT):")
        for metric in ["accuracy", "mean_similarity"]:
            avg_by_type = category_df.pivot(
                index="reasoning_type", columns="model", values=metric
            )
            gaps = (avg_by_type["Gemini"] - avg_by_type["GPT"]).sort_values(ascending=False)

            print(f"\n{metric.replace('_', ' ').title()} Gaps:")
            print("\nGemini Advantage:")
            for type_, gap in gaps.head(3).items():
                print(f"- {type_}: {gap:.4f}")
            print("\nGPT Advantage:")
            for type_, gap in gaps.tail(3).items():
                print(f"- {type_}: {gap:.4f}")


def main():
    # Load results
    print("Loading results...")
    gemini_results = load_results("results/gemini_results.txt")
    gpt_results = load_results("results/gpt_results.txt")
    
    print("\nPreparing data...")
    # Prepare data
    df = prepare_data(gemini_results, gpt_results)
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())
    print("\nDataFrame Columns:", df.columns.tolist())
    
    if len(df) == 0:
        print("Error: No data was loaded into the DataFrame")
        return
        
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    print("Visualization has been saved as 'combined_accuracy_comparison.png', 'combined_mean_similarity_comparison.png', 'individual_accuracy_comparison.png', and 'individual_mean_similarity_comparison.png'")

    # Analyze performance
    print("\nAnalyzing performance...")
    analyze_performance(df)

    # Print some statistics
    print("\nNumber of data points:")
    print(df.groupby("model").size())


if __name__ == "__main__":
    main()
