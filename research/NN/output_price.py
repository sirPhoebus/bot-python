import json
import requests
import pandas as pd

symbol = 'BTCUSDT'
tf = '1d'

# Set the API endpoint URL
url = 'https://api.binance.com/api/v3/klines'

# Set the request parameters
# "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
params = {
    'symbol': symbol,
    'interval': tf ,
    'limit': 1000
}

# Make the request to the API
response = requests.get(url, params=params)

# Check the status code to make sure the request was successful
if response.status_code != 200:
    raise ValueError('Failed to get data')

# Convert the response to a JSON object
data = response.json()

# Convert the price data to the desired format and write it to a JSON file
inputs = []
targets = []
for row in data:
    inputs.append([float(row[1]), float(row[4])])
    targets.append(float(row[4]))

# Create a list of dictionaries
data_list = []
for i in range(len(inputs)):
    data_list.append({
        'inputs': inputs[i],
        'targets': [targets[i]]
    })

# Write the data to a JSON file
with open('data.json', 'w') as f:
    json.dump({'data': data_list}, f)
