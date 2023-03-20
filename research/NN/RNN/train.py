import torch
import torch.nn as nn
import numpy as np
from data import download_data, preprocess_data
from datetime import datetime

st = '1 Jan 2020'

input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 100

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
    
# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
startDate = int(datetime.strptime(st, '%d %b %Y').timestamp())
data = download_data('BTCUSDT', '1h', startDate)
data = preprocess_data(data.values)
print('data loaded.')
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
train_data_tensor = torch.FloatTensor(train_data).reshape(-1, input_size)
net = RNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i in range(input_size, len(train_data_tensor)):
        # Get the input and target sequences
        inputs = train_data_tensor[i-input_size:i]
        target = train_data_tensor[i:i+output_size]
        # Move the input and target sequences to the device
        inputs = inputs.to(device)
        target = target.to(device)
        # Forward pass
        outputs = net(inputs.unsqueeze(0))
        loss = criterion(outputs, target)
        print(f'Epoch: {epoch+1}/{num_epochs} Loss: {loss.item():.4f}')
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print the loss after every epoch
print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
torch.save(net.state_dict(), 'rnn.pth')

