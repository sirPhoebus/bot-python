import torch
import numpy as np
from data import download_data
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1
st = '1 Jan 2020'
scaler = MinMaxScaler(feature_range=(0, 1))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Function to preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

startDate = int(datetime.strptime(st, '%d %b %Y').timestamp())
data = download_data('BTCUSDT', '1h', startDate)
data = preprocess_data(data.values)
print('data loaded.')


# Load the model
net = RNN(input_size, hidden_size, num_layers, output_size).to(device)
net.load_state_dict(torch.load('rnn.pth'))

# Prepare test data tensor
test_size = len(data) - int(len(data) * 0.7)
test_data = data[-test_size:]
test_data_tensor = torch.FloatTensor(test_data).reshape(-1, input_size)

# Make predictions on test data
predictions = []
targets = []
with torch.no_grad():
    for i in range(input_size, len(test_data_tensor)):
        # Get the input and target sequences
        inputs = test_data_tensor[i-input_size:i]
        target = test_data_tensor[i:i+output_size]

        # Move the input and target sequences to the device
        inputs = inputs.to(device)
        target = target.to(device)

        # Forward pass
        outputs = net(inputs.unsqueeze(0))

        # Append the prediction and target to the lists
        predictions.append(outputs.cpu().numpy())
        targets.append(target.cpu().numpy())


# Concatenate the predictions and targets
predictions = np.concatenate(predictions, axis=0)
targets = np.concatenate(targets, axis=0)

# Fit the scaler on the test data
scaler.fit(test_data)
print(predictions)
# Inverse transform the predictions and targets
predictions = scaler.inverse_transform(predictions)
targets = scaler.inverse_transform(targets)

# Plot the results
plt.plot(predictions, label='Predictions')
plt.plot(targets, label='Targets')
plt.legend()
plt.show()