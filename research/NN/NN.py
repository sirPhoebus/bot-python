import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
# Define command-line arguments
parser = argparse.ArgumentParser(description='Train neural network on input data')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
args = parser.parse_args()

# Extract the value of num_epochs from the command-line arguments
num_epochs = args.num_epochs


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
data_loader = DataLoader(dataset, batch_size=24, shuffle=True)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(24, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 24)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network and move it to the GPU
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
optimizer = torch.optim.Adamax(net.parameters(), lr=0.001)

if os.path.isfile('./NN.pth'):
    print("Loading model...")
    checkpoint = torch.load('NN.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Model loaded.")
    net = Net().to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
else: 
    # Train the neural network on the GPU
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch+1}/{num_epochs} Loss: {loss.item():.4f}')
    # Save the state of the model and optimizer
    print("Saving new model...")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'NN.pth')
    test_loader = DataLoader(dataset, batch_size=24, shuffle=True)
    # Evaluate the trained neural network on the input data
    # Initialize counters
    correct = 0
    total = 0
    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Make predictions
            outputs = net(inputs)
            predicted = outputs.argmax(dim=1)  # get the index of the maximum value in the output tensor
            
            # Update counters
            total += targets.size(0)
            correct += (predicted == targets.argmax(dim=1)).sum().item()
    # Compute accuracy
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
