import torch
import torch.nn as nn
from torchview import draw_graph

# Define your LinkPredictor model
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

# Create an instance of your model
input_size = 300  # Example input size
model = LinkPredictor(input_size)

# Visualize the model graph
model_graph = draw_graph(
    model,
    input_size=(1, input_size),  # Adjust the input size as needed
    expand_nested=True
)

# Display the graph
model_graph.visual_graph.render("link_predictor_model_graph", format="pdf")


