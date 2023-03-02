import json
import requests
import pandas as pd

symbol = 'BTCUSDT'
tf = '1h'

# Set the API endpoint URL
url = 'https://api.binance.com/api/v3/klines'

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

# Convert the response to a Pandas dataframe
df = pd.DataFrame(response.json(), columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert the timestamp to a datetime object
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Slice the data into 24-hourly candles
hourly_data = df.groupby(pd.Grouper(key='open_time', freq='24H'))
# Print the number of groups
print(f'Number of groups: {len(hourly_data.groups)}')
# Print the data for each group
# for group_name, group_data in hourly_data:
#     print(f'Group name: {group_name}')
#     print(group_data)
# Convert the price data to the desired format and write it to a JSON file
data_list = []
idx = 0 
for idx, group in enumerate(hourly_data):
    inputs = []
    targets = []
    for row in group[1].itertuples():
        inputs.append(float(row.open))
        targets.append(float(row.close))
    data_list.append({
        #'id': idx + 1,
        'inputs': inputs,
        'targets': targets
    })

# Write the data to a JSON file
with open('data.json', 'w') as f:
    json.dump({'data': data_list}, f)
print('Json created.')

# As 1000 is not divisible by 24...the output does not contains 24 values in the starting of the file and at the end either 
# So either delete the occurences or adapt the code.