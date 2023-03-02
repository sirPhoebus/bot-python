import requests
import pandas as pd
import csv

symbol = 'BTCUSDT'
tf = '1d'
url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': symbol,
    'interval': tf,
    'limit': 1000
}

response = requests.get(url, params=params)
if response.status_code != 200:
    raise ValueError('Failed to get data')

df = pd.DataFrame(response.json(), columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df.set_index('open_time', inplace=True)

with open('data.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Date', 'Close'])
    for row in df.itertuples():
        if pd.notnull(row.close):
            csv_writer.writerow([row.Index.strftime('%Y-%m-%d'), row.close])
    print('CSV file created.')