# -*- coding: utf-8 -*-
"""GNN(small_dataset_Cora).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uRolJi1a7BOG53p5Aes5_X5VgaDSsUaA
"""

!pip install torch-geometric
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

# Load the Cora dataset (small and fast)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move dataset to GPU
data = data.to(device)

# Initialize the model, optimizer, and loss function
model = GraphSAGE(dataset.num_features, 128, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Create NeighborLoader for training with optimized settings
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # Smaller neighbor samples for speed
    batch_size=128,  # Smaller batch size for quick training
    shuffle=True,
    input_nodes=data.train_mask
)

# Lists to store loss and accuracy
train_losses = []
test_accuracies = []

# Training loop
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)  # Move batch to GPU
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    return total_loss / len(train_loader.dataset)

# Evaluation function
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    return correct / data.test_mask.sum().item()

# Training and testing loop
num_epochs = 20
for epoch in range(num_epochs):
    loss = train()
    train_losses.append(loss)

    if epoch % 2 == 0:
        acc = test()
        test_accuracies.append(acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# Final test accuracy
acc = test()
print(f'Final Test Accuracy: {acc:.4f}')

# Plot training loss and test accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(range(0, num_epochs, 2), test_accuracies, marker='s', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Over Epochs")

plt.tight_layout()
plt.show()