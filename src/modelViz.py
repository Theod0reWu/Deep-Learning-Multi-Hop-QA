import torch
import torch.nn as nn
from torchviz import make_dot

# Define the LinkPredictor model
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

# Create an instance of the model
input_size = 300  # Example input size
model = LinkPredictor(input_size)

# Generate a dummy input tensor
dummy_input = torch.randn(1, input_size)

# Perform a forward pass to visualize the computation graph
output = model(dummy_input)

# Create the computational graph using make_dot
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PDF file
graph.render("link_predictor_computation_graph", format="pdf")

