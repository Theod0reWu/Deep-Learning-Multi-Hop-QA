import matplotlib.pyplot as plt
import numpy as np
import re

# Helper function to parse results.txt
def parse_results(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Extract metrics using regex
    models = re.findall(r'Model: (.*?)\n.*?total_queries: (\d+).*?avg_retrieved_docs: ([\d.]+).*?avg_retrieval_iterations: ([\d.]+).*?avg_answer_similarity: ([\d.]+).*?accuracy_rate: ([\d.]+)', data, re.DOTALL)

    parsed_results = []
    for model in models:
        parsed_results.append({
            'model': model[0],
            'total_queries': int(model[1]),
            'avg_retrieved_docs': float(model[2]),
            'avg_retrieval_iterations': float(model[3]),
            'avg_answer_similarity': float(model[4]),
            'accuracy_rate': float(model[5])
        })
    return parsed_results

# Function to calculate NAEI and prepare data for plotting
def calculate_naei(data):
    results = []
    for entry in data:
        accuracy = entry['accuracy_rate']
        token_usage = entry['avg_retrieved_docs']  # Token usage per hop (retrieved docs)
        hop_count = entry['avg_retrieval_iterations']
        
        naei = accuracy / (token_usage * hop_count)
        results.append({'model': entry['model'], 'hop_count': hop_count, 'naei': naei})
    return results

# Function to plot NAEI vs Hop Count for specific models
def plot_naei(naei_data):
    allowed_models = [
        "predicted hops",
        "predicted hops + 1",
        "predicted hops + 2",
        "predicted hops + 3",
    ]

    plt.figure(figsize=(10, 6))
    for model in allowed_models:
        model_data = [(entry['hop_count'], entry['naei']) for entry in naei_data if model in entry['model']]
        if model_data:
            model_data.sort()
            hop_counts, naei_values = zip(*model_data)
            plt.plot(hop_counts, naei_values, marker='o', label=model)
    
    plt.xlabel('Hop Count')
    plt.ylabel('NAEI (Normalized Accuracy Efficiency Index)')
    plt.title('NAEI vs Hop Count (Predicted Hops Variants)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Path to results.txt
    results_file = 'results.txt'

    # Parse results and calculate NAEI
    parsed_data = parse_results(results_file)
    naei_data = calculate_naei(parsed_data)

    # Generate and display the plot
    plot_naei(naei_data)
