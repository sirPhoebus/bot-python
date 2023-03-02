import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import NN as nn
# Define command-line arguments
parser = argparse.ArgumentParser(description='Predict target value using trained neural network')
parser.add_argument('--input_data', type=float, nargs=24, default=[0.0]*24, help='Input data for prediction')
args = parser.parse_args()

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained neural network from the saved checkpoint file
checkpoint = torch.load('NN.pth')
net = nn.Net()
net.load_state_dict(checkpoint['model_state_dict'])

# Prepare the input data as a PyTorch tensor
input_data = torch.tensor(args.input_data, dtype=torch.float32).to(device)

# Pass the input data through the trained neural network to get the predicted output
with torch.no_grad():
    output = net(input_data)
    predicted_target = output.item()

print(f'Predicted target value: {predicted_target:.4f}')
