import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load input data from a JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Convert input data to PyTorch tensors and create a DataLoader object
inputs = torch.tensor([d['inputs'] for d in data['data']], dtype=torch.float32).to(device)
targets = torch.tensor([d['targets'] for d in data['data']], dtype=torch.float32).to(device)

dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network and move it to the GPU
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Define hyperparameters
num_epochs = 10

# Train the neural network on the GPU
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
# Evaluate the trained neural network on the input data
with torch.no_grad():
    net.eval()
    inputs = torch.tensor([d['inputs'] for d in data['data']], dtype=torch.float32).to(device)
    targets = torch.tensor([d['targets'] for d in data['data']], dtype=torch.float32).to(device)
    outputs = net(inputs)

    # Calculate the mean squared error between the predicted outputs and the actual targets
    mse = criterion(outputs, targets)

    # Print the mean squared error and the predicted outputs
    print(f'MSE: {mse:.4f}')
    for i in range(len(outputs)):
        print(f'Input: {inputs[i].tolist()}, Target: {targets[i].item():.1f}, Predicted: {outputs[i].item():.1f}')
