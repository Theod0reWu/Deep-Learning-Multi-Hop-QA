import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from .model import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from .dataset import get_frames_filtereddataset

def visualize_predictions(true_values, predicted_values, save_path=None):
    """
    Create visualization plots comparing predicted vs true values.
    
    Args:
        true_values: Array of ground truth values
        predicted_values: Array of model predictions
        save_path: Optional path to save the plot instead of displaying
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot with perfect prediction line
    plt.subplot(2, 2, 1)
    plt.scatter(true_values, predicted_values, alpha=0.5, c='blue', label='Predictions')
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Number of Wiki Links')
    plt.ylabel('Predicted Number of Wiki Links')
    plt.title('Predicted vs True Values\n(closer to red line is better)')
    plt.legend()
    
    # 2. Distribution of prediction errors
    plt.subplot(2, 2, 2)
    errors = predicted_values - true_values
    sns.histplot(errors, kde=True, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors\n(centered at 0 is better)')
    plt.legend()
    
    # 3. Box plot of predictions for each true value
    plt.subplot(2, 2, 3)
    data = pd.DataFrame({'True Values': true_values, 'Predicted Values': predicted_values})
    sns.boxplot(x='True Values', y='Predicted Values', data=data)
    plt.xlabel('True Number of Wiki Links')
    plt.ylabel('Predicted Number of Wiki Links')
    plt.title('Distribution of Predictions per True Value\n(smaller boxes = more consistent)')
    
    # 4. Additional statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""Prediction Statistics:

    Mean Absolute Error: {np.mean(np.abs(errors)):.2f}
    Mean Error: {np.mean(errors):.2f}
    Std Dev of Error: {np.std(errors):.2f}
    
    Min Error: {np.min(errors):.2f}
    Max Error: {np.max(errors):.2f}
    
    % Within ±0.5: {100 * np.mean(np.abs(errors) <= 0.5):.1f}%
    % Within ±1.0: {100 * np.mean(np.abs(errors) <= 1.0):.1f}%
    % Within ±2.0: {100 * np.mean(np.abs(errors) <= 2.0):.1f}%
    
    Total Predictions: {len(true_values)}
    
    Range of True Values: {min(true_values):.0f} to {max(true_values):.0f}
    Range of Predictions: {min(predicted_values):.1f} to {max(predicted_values):.1f}
    """
    plt.text(0.1, 0.1, stats_text, fontsize=12, family='monospace')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def analyze_model(model_dir='models', batch_size=32):
    """Load a trained model and analyze its predictions"""
    # Load model and vectorizer
    print("Loading model...")
    model, vectorizer = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get dataset
    print("Loading dataset...")
    df = get_frames_filtereddataset()
    
    # Prepare data
    print("Preparing predictions...")
    prompts = list(df["Prompt"])
    true_values = np.array(df["query_count"], dtype=np.float32)
    
    # Vectorize text
    X = vectorizer.transform(prompts)
    X = np.array(X.todense())
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch)
            predictions.extend(outputs.squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Visualize results
    print("\nGenerating visualization plots...")
    visualize_predictions(true_values, predictions)
    
    return true_values, predictions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze trained model predictions')
    parser.add_argument('--model-dir', default='models', help='Directory containing saved model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for predictions')
    args = parser.parse_args()
    
    analyze_model(args.model_dir, args.batch_size)
