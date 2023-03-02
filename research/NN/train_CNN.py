import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import torch.nn.functional as F

# Define command-line arguments
parser = argparse.ArgumentParser(description='Train neural network on input data')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
args = parser.parse_args()

# Extract the value of num_epochs from the command-line arguments
num_epochs = args.num_epochs

# Define hyperparameters
# num_epochs = 100
# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load input data from a JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Extract inputs and targets from the data dictionary
inputs = torch.tensor([d['inputs'] for d in data['data']], dtype=torch.float32)
targets = torch.tensor([d['targets'] for d in data['data']], dtype=torch.float32)
inputs = inputs[:-(inputs.shape[0] % 24)]
targets = targets[:-(targets.shape[0] % 24)]

inputs = inputs.to(device)
targets = targets.to(device)

# Move inputs and targets to the GPU if available

dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=24, shuffle=True)

# Define the neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # define max pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # define fully connected layers
        self.fc1 = nn.Linear(128*3, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # apply convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # flatten output from convolutional layers for input to fully connected layers
        x = x.view(-1, 128*3)
        
        # apply fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Create an instance of the neural network and move it to the GPU
net = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# To resume training later, you can load the saved state using the `torch.load` function
#CHeck if there is a saved model to resume training from
if os.path.isfile('./checkpoint.pth'):
    print("Loading model...")
    checkpoint = torch.load('checkpoint.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# Train the neural network on the GPU
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(epoch)

# Save the state of the model and optimizer
print("Saving new model...")
torch.save({
    'epoch': num_epochs,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'checkpoint.pth')
