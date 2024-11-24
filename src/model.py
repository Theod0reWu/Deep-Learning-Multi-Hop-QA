import torch
import torch.nn as nn

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

def save_model(model, vectorizer, save_dir):
    """Save model weights and vectorizer to files"""
    import os
    import pickle
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save vectorizer
    vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model(save_dir, input_size=300):
    """Load saved model weights and vectorizer"""
    import os
    import pickle
    
    # Load model weights
    model = LinkPredictor(input_size)
    model_path = os.path.join(save_dir, 'model_weights.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load vectorizer
    vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer
