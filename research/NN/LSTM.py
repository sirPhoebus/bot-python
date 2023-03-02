import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load the BTC price data
df = pd.read_csv('data.csv')
df = df.iloc[::-1].reset_index(drop=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
# Define the training data
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
train_data_tensor = torch.FloatTensor(train_data).view(-1)

# Define the test data
test_data = data[train_size:]
test_data_tensor = torch.FloatTensor(test_data).view(-1)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Set the hyperparameters
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.0001
num_epochs = 10000

# Create an instance of the LSTM model and move it to the GPU
lstm = LSTM(input_size, hidden_size, num_layers, output_size).cuda()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the LSTM model on the GPU
for epoch in range(num_epochs):
    # Forward pass
    train_data_input = Variable(train_data_tensor[:-1].view(-1, 1, input_size)).cuda()
    train_data_target = Variable(train_data_tensor[1:].view(-1, 1)).cuda()
    output = lstm(train_data_input)

    # Compute the loss
    loss = criterion(output, train_data_target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at every epoch
    print(f'Epoch: {epoch+1}/{num_epochs} Loss: {loss.item():.4f}')

# Evaluate the LSTM model on the test data
lstm.eval()
test_data_input = Variable(test_data_tensor[:-1].view(-1, 1, input_size)).cuda()
with torch.no_grad():
    test_output = lstm(test_data_input)

# Convert the predicted test data back to its original scale
test_output = scaler.inverse_transform(test_output.cpu().numpy())

# Print the predicted test data
print('Predicted values:')
for i in range(len(test_output)):
    print(f'{i+1}: {test_output[i][0]:.4f}')