
import time
import requests
import pandas as pd
import datetime
import numpy as np 


def generate_trend():
    # Initialize variables
    trend = None
    pd.options.display.max_rows = 5
    mult = 2
    length = 20
    def getHistorical():
    # Set the API endpoint URL
        url = 'https://api.binance.com/api/v3/klines'

        # Set the request parameters
        # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'limit': '10' # max 1000
        }

        # Make the request to the API
        response = requests.get(url, params=params)

        # Check the status code to make sure the request was successful
        # Convert the response to a JSON object
        df = response.json()    
        return df

    while True:
        df = pd.DataFrame(getHistorical(), columns=[
        "open_time",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "unused_field"
        ])
        df["open_time"] = df["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
        # Convert the close_price, low_price, and high_price columns to float
        df["close_price"] = df["close_price"].astype(float)
        df["low_price"] = df["low_price"].astype(float)
        df["high_price"] = df["high_price"].astype(float)

        current_price = df.iloc[-1]["close_price"]
        hp = df["high_price"]
        basis = np.mean(hp[-length:])
        dev = mult * np.std(hp[-length:])
        upper = basis + dev
        lower = basis - dev
        if current_price <= lower:
            print('Lower Bound Hit!')
        if current_price >= upper:
            print('Upper Bound Hit!')
        if current_price > lower or current_price < upper:
            print('In range')

        time.sleep(1)
        
generate_trend()

