import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(self, n_features, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(n_features, dropout)
        encoder_layers = nn.TransformerEncoderLayer(n_features, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x

# Define the positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Define a custom dataset to load the stock prices
class StockDataset(Dataset):
    def __init__(self, prices, seq_length):
        self.prices = prices
        self.seq_length = seq_length

    def __len__(self):
        return len(self.prices) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.prices[idx:idx+self.seq_length]
        y = self.prices[idx+self.seq_length]
        return x, y

# Load the stock prices and preprocess them
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-01-01'
data = yf.download(symbol, start=start_date, end=end_date)
prices = data['Adj Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices = scaler.fit_transform(prices)

# Split the data into training and test sets
seq_length = 32
train_size = int(0.8 * len(prices))
train_prices = prices[:train_size]
test_prices = prices[train_size-seq_length:]

# Create the dataloaders
train_dataset = StockDataset(train_prices, seq_length)
test_dataset = StockDataset(test_prices, seq_length)
train_dataloader = DataLoader(train_dataset, batch_size=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_features = 8
nhead = 8
nhid = 256
nlayers = 6
lr = 0.0001
num_epochs = 50

model = TransformerModel(n_features, nhead, nhid, nlayers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss/len(train_dataloader)))
model.eval()
test_loss = 0.0
predictions = []
with torch.no_grad():
    for inputs, labels in train_dataloader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predictions.append(outputs.cpu().numpy())
print('Test Loss: {:.4f}'.format(test_loss/len(train_dataloader)))

predictions = np.concatenate(predictions)
predictions = scaler.inverse_transform(predictions)
test_prices = scaler.inverse_transform(test_prices)
plt.plot(test_prices, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()