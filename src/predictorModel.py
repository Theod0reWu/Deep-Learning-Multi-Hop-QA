import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.dataset import get_frames_filtereddataset
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Fix SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class PromptDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.vectors[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class LinkPredictor(nn.Module):
    def __init__(self, input_size):
        super(LinkPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Lower dropout rate

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def calculate_metrics(outputs, labels):
    # Convert lists to numpy arrays
    pred = np.array(outputs)
    true = np.array(labels)

    # Calculate MAE
    mae = np.mean(np.abs(pred - true))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    # Calculate "accuracy" (predictions within 0.5 of true value)
    accuracy = np.mean(np.abs(pred - true) <= 0.5)

    return {
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy * 100
    }


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


def train_model(num_epochs=50, batch_size=32, learning_rate=0.001):
    # Get dataset
    print("Loading dataset...")
    df = get_frames_filtereddataset()

    # Prepare data and debug types
    print("Preparing data...")
    prompts = list(df["Prompt"])
    print(f"Type of prompts: {type(prompts)}")
    print(f"Type of first prompt: {type(prompts[0])}")
    print(f"Number of prompts: {len(prompts)}")
    
    labels = np.array(df["query_count"], dtype=np.float32)
    print(f"Type of labels: {type(labels)}")
    print(f"Shape of labels: {labels.shape}")

    # Create TF-IDF vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(prompts)
    print(f"Type of X: {type(X)}")
    print(f"Shape of X: {X.shape}")
    X = np.array(X.todense())

    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = PromptDataset(X_train, y_train)
    val_dataset = PromptDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LinkPredictor(300)  # 300 features from TF-IDF
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_true_labels = []

        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), labels.float())
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(), labels.float())

                val_loss += loss.item()
                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_metrics = calculate_metrics(train_predictions, train_true_labels)
        val_metrics = calculate_metrics(val_predictions, val_true_labels)

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print("Training metrics:")
        print(f"  Loss: {train_loss/len(train_loader):.4f}")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  Accuracy (±0.5): {train_metrics['accuracy']:.2f}%")
        print("Validation metrics:")
        print(f"  Loss: {val_loss/len(val_loader):.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Accuracy (±0.5): {val_metrics['accuracy']:.2f}%")

    # Visualize final predictions
    print("\nGenerating prediction distribution plots...")
    visualize_predictions(np.array(val_true_labels), np.array(val_predictions))
    
    return model, vectorizer

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib", "seaborn"])
        import matplotlib.pyplot as plt
        import seaborn as sns
        
    model, vectorizer = train_model(num_epochs=50, batch_size=32, learning_rate=0.001)
