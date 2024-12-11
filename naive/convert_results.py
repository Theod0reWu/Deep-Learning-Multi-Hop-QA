def parse_text_metrics(content):
    metrics = {}
    current_type = None
    overall_metrics = {}

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
                    overall_metrics[key] = value
                else:
                    if current_type not in metrics:
                        metrics[current_type] = {}
                    metrics[current_type][key] = value
            elif not any(
                metric in line.lower() for metric in ["accuracy:", "mean_similarity:"]
            ):
                current_type = line.split(":")[0].strip()

    return overall_metrics, metrics


def format_json_output(overall_metrics, metrics):
    output = []
    output.append("Evaluation Results")
    output.append("=================")
    output.append("")
    output.append(
        "Overall GPT Metrics (using sentence-transformer for embedding similarity):"
    )
    output.append("--------------")
    output.append("{")
    output.append(f'  "accuracy": {overall_metrics.get("accuracy", 0)},')
    output.append(f'  "mean_similarity": {overall_metrics.get("mean_similarity", 0)}')
    output.append("}")
    output.append("")
    output.append("Metrics by Reasoning Type:")
    output.append("------------------------")
    output.append("{")

    # Convert metrics to JSON format
    metric_lines = []
    for reasoning_type, values in metrics.items():
        metric_lines.append(f'  "{reasoning_type}": {{')
        metric_lines.append(f'    "accuracy": {values.get("accuracy", 0)},')
        metric_lines.append(
            f'    "mean_similarity": {values.get("mean_similarity", 0)}'
        )
        metric_lines.append(
            "  }" + ("," if reasoning_type != list(metrics.keys())[-1] else "")
        )

    output.extend(metric_lines)
    output.append("}")

    return "\n".join(output)


def main():
    # Read the current GPT results
    with open("results/gpt_results.txt", "r") as f:
        content = f.read()

    # Parse and convert to new format
    overall_metrics, metrics = parse_text_metrics(content)
    new_content = format_json_output(overall_metrics, metrics)

    # Write the new format
    with open("results/gpt_results_new.txt", "w") as f:
        f.write(new_content)

    print("Converted GPT results saved to 'results/gpt_results_new.txt'")


if __name__ == "__main__":
    main()
