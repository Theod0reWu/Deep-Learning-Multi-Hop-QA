import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import get_frames_filtereddataset
import ssl

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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def calculate_metrics(outputs, labels):
    # Convert to numpy for easier calculation
    pred = outputs.cpu().detach().numpy()
    true = labels.cpu().detach().numpy()

    # Calculate MAE
    mae = np.mean(np.abs(pred - true))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    # Calculate "accuracy" (predictions within 0.5 of true value)
    accuracy = np.mean(np.abs(pred - true) <= 0.5)

    return mae, rmse, accuracy


def train_model(num_epochs=10, batch_size=32, learning_rate=0.001):
    # Get dataset
    print("Loading dataset...")
    df = get_frames_filtereddataset()

    # Prepare data
    prompts = df["Prompt"].values
    labels = df["query_count"].values

    # Create TF-IDF vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(prompts).toarray()

    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = PromptDataset(X_train, y_train)
    val_dataset = PromptDataset(X_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LinkPredictor(300).to(device)  # 300 features from TF-IDF

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_mae = train_rmse = train_acc = 0
        num_batches = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mae, rmse, acc = calculate_metrics(outputs.squeeze(), labels)
            train_mae += mae
            train_rmse += rmse
            train_acc += acc
            num_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_mae = val_rmse = val_acc = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                mae, rmse, acc = calculate_metrics(outputs.squeeze(), labels)
                val_mae += mae
                val_rmse += rmse
                val_acc += acc
                val_batches += 1

        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Training metrics:")
        print(f"  Loss: {total_loss/num_batches:.4f}")
        print(f"  MAE: {train_mae/num_batches:.4f}")
        print(f"  RMSE: {train_rmse/num_batches:.4f}")
        print(f"  Accuracy (±0.5): {train_acc/num_batches*100:.2f}%")
        print(f"Validation metrics:")
        print(f"  Loss: {val_loss/val_batches:.4f}")
        print(f"  MAE: {val_mae/val_batches:.4f}")
        print(f"  RMSE: {val_rmse/val_batches:.4f}")
        print(f"  Accuracy (±0.5): {val_acc/val_batches*100:.2f}%")

    return model, vectorizer


if __name__ == "__main__":
    model, vectorizer = train_model()
