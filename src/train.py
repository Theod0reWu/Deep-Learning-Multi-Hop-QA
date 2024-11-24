import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from .dataset import get_frames_filtereddataset
from .model import LinkPredictor, save_model
import os

class PromptDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input": self.X[idx],
            "label": self.y[idx]
        }

def calculate_metrics(outputs, labels):
    """Calculate MAE, RMSE, and accuracy metrics"""
    pred = np.array(outputs)
    true = np.array(labels)
    
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    accuracy = np.mean(np.abs(pred - true) <= 0.5)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy * 100
    }

def train_model(num_epochs=50, batch_size=32, learning_rate=0.001, save_dir='models'):
    """Train the model and save weights"""
    # Get dataset
    print("Loading dataset...")
    df = get_frames_filtereddataset()

    # Prepare data
    print("Preparing data...")
    prompts = list(df["Prompt"])
    labels = np.array(df["query_count"], dtype=np.float32)

    # Create TF-IDF vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(prompts)
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

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, vectorizer, save_dir)

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

    print(f"\nTraining complete. Model saved to {save_dir}")
    return val_predictions, val_true_labels

if __name__ == "__main__":
    train_model()
