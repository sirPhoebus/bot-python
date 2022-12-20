
import time
import requests
import pandas as pd
import json

# Initialize variables
prev_price = None
trend = None

def getHistorical():
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': '10' 
    }

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check the status code to make sure the request was successful
    # Convert the response to a JSON object
    data = response.json()    
    return data

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
    print(df)
    # Calculate the moving average of the last 5 days
    moving_avg = float(df["close_price"].apply(float).mean())
    current_price = float(df.tail(1)["close_price"])
    # Check if the current price is higher or lower than the moving average
    if current_price > moving_avg:
        trend = "uptrend"
    elif current_price < moving_avg:
        trend = "downtrend"
    else:
        trend = "neutral"

    # Print the current price and trend
    print(f"Current MA: {round(moving_avg, 1)}")
    print(f"Current price: {round(current_price, 1)} ({trend})")

    # Sleep for 1 minute
    time.sleep(60)
