
import time
import requests
import pandas as pd
import datetime
import talib 
import numpy as np 

# Initialize variables
prev_price = None
trend = None
pd.options.display.max_rows = 50

def getHistorical():
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '15m',
        'limit': '50' # max 1000
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

    # Calculate the moving average
    moving_avg = float(df["close_price"].apply(float).mean())
    current_price = float(df.tail(1)["close_price"])
    cp = df["close_price"]
    lp = df["low_price"]
    hp = df["high_price"]

    # Calculate the Average True Range(ATR)
    # The value of the ATR is not directly related to the price of an asset. Instead, it reflects the degree of price fluctuation over a given time period. 
    # A higher ATR value indicates that the asset has had a larger range of price movements over the given time period, while a lower ATR value indicates a smaller range of price movements.
    df['ATR'] = talib.ATR(hp, lp, cp, timeperiod=24)
    # Calculate the rolling mean of ATR
    df['ATR_MA_4'] = df['ATR'].rolling(4).mean()
    # Flag the minutes where ATR breaks out its rolling mean
    df['ATR_breakout'] = np.where((df['ATR'] > df['ATR_MA_4']), True, False)
    # Calculate the three-candle rolling High
    df['three_candle_High'] = hp.rolling(3).max()
    # Check if the fourth candle is Higher than the Highest of the previous 3 candle
    df['four_candle_High'] = np.where( hp >
        df['three_candle_High'].shift(1), True, False)
    # Calculate the three-candle rolling Low
    df['three_candle_Low'] = lp.rolling(3).min()
    # Check if the fourth candle is Lower than the Lowest of the previous 3 candles
    df['four_candle_Low'] = np.where( lp <
        df['three_candle_Low'].shift(1), True, False)
    # Flag long positions
    df['long_positions'] = np.where(df['ATR_breakout'] & df['four_candle_High'], 1, 0)
    # Flag short positions
    df['short_positions'] = np.where(df['ATR_breakout'] & df['four_candle_Low'], -1, 0)
    # Combine long and short  position flags
    df['positions'] = df['long_positions'] + df['short_positions']
    print(df)


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
