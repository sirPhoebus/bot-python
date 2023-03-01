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

# Extract inputs and targets from the data dictionary
inputs = torch.tensor([d['inputs'] for d in data['data']], dtype=torch.float32)
targets = torch.tensor([d['targets'] for d in data['data']], dtype=torch.float32)
inputs = inputs[:-(inputs.shape[0] % 32)]
targets = targets[:-(targets.shape[0] % 32)]

inputs = inputs.to(device)
targets = targets.to(device)

# Move inputs and targets to the GPU if available

dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=512, out_features=128) # Update out_features to match the input size
        self.fc2 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x):
        x = x.unsqueeze(1) # Add a channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Create an instance of the neural network and move it to the GPU
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Define hyperparameters
num_epochs = 1000

# Train the neural network on the GPU
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(epoch)

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
