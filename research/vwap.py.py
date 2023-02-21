
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
        if df.empty:
            print("Error: Empty DataFrame")
            break
        if df.isna().sum().sum() > 0:
            print("Error: Missing values in DataFrame")
            break
        df["open_time"] = df["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
        # Convert the close_price, low_price, and high_price columns to float
        df["close_price"] = df["close_price"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Define the period for calculating the VWAP
        period = 14

        # Create a new column for the cumulative volume
        df['cum_volume'] = df['volume'].cumsum()
        df["cum_volume"] = df["cum_volume"].astype(float)
        # Create a new column for the VWAP produces NaN
        # df['VWAP'] = (df['cum_volume'] - df['volume'].shift(period)) / (df['cum_volume'] - df['cum_volume'].shift(period)) * df['close_price']
        df['VWAP'] = (df['cum_volume'] - df['volume'].rolling(period).sum() ) / (df['cum_volume'] - df['cum_volume'].rolling(period).sum() ) * df['close_price']
        # Print the DataFrame
        print(df)

        time.sleep(1)
        
generate_trend()

