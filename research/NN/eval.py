

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import train_CNN as t
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


# Create an instance of the neural network and move it to the GPU
net = t.CNN().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# To resume training later, you can load the saved state using the `torch.load` function
#CHeck if there is a saved model to resume training from
print("Loading model...")
checkpoint = torch.load('checkpoint.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Evaluate the trained neural network on the input data
with torch.no_grad():
    net.eval()
    outputs = net(inputs)

    # Calculate the mean squared error between the predicted outputs and the actual targets
    mse = criterion(outputs, targets)

    # Print the mean squared error and the predicted outputs
    print(f'MSE: {mse:.4f}')
    for i in range(len(outputs)):
        print(f'Input: {inputs[i].tolist()}, Target: {targets[i].item():.1f}, Predicted: {outputs[i].item():.1f}')
