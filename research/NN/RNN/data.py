import pandas as pd
import numpy as np
import requests
from datetime import datetime

st = '1 Jan 2020'
end = '31 Dec 2020'

# Function to download data from the Binance API
def download_data(symbol, interval, start_date):

    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_date

        
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError('Failed to get data')
    df = pd.DataFrame(response.json(), columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df




# startDate = int(datetime.strptime(st, '%d %b %Y').timestamp())
# endDate = int(datetime.strptime(end, '%d %b %Y').timestamp())
# df = download_data('BTCUSDT', '1d', startDate)
# print(preprocess_data(df.values))
    
