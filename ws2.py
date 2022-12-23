import unicorn_binance_websocket_api
import websocket
import json
import pandas as pd
import datetime
import talib 
import numpy as np 
import termcolor
import os
import requests

if not os.path.exists('log.txt'):
  # Create the file if it does not exist
  open('log.txt', 'w').close()
# Create a variable to keep track of the current position (0 = no position, 1 = long position, -1 = short position)
position = 0
risk = 0.95
# Create a variable to keep track of the portfolio value
portfolio_value = 10000
SL = 0


# Load the historical data for the asset
def getHistorical():
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': '15' # max 1000
    }

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check the status code to make sure the request was successful
    # Convert the response to a JSON object
    df = response.json()    
    return df

def process_message(msg):
    """Processes a websocket message and adds the close price to the dataframe."""
    data = json.loads(msg)
    k_dict = data['data']['k']
    df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
    df['close_time'] = pd.to_datetime(df['t'], unit='ms')
    df['close_price'] = df['c'].astype(float)
    df = df[['close_time', 'close_price']]
    return df

def calculate_speeds(df):
    """Calculates the speeds per second from the data in the dataframe."""
    # Extract the close prices from the dataframe
    close_prices = df['close_price'].values
    # Calculate the speeds as the difference between consecutive close prices
    speeds = np.diff(close_prices)
    # Return the speeds per second
    return speeds

def calculate_average_acceleration(speeds, time_intervals):
    """Calculates the average acceleration from the speeds and time intervals."""
    accelerations = []
    for i in range(1, len(speeds)):
        acceleration = (speeds[i] - speeds[i-1]) / time_intervals[i]
        accelerations.append(acceleration)
    return sum(accelerations) / len(accelerations)

# Connect to the websocket API and create a stream
ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1m'], ['btcusdt'])

# Create an empty dataframe to store the close prices
df = pd.DataFrame(columns=['close_time', 'close_price'])
position = 0

skip_first_message = True
while True:
    data_api = pd.DataFrame(getHistorical(), columns=[
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
    data_api["open_time"] = data_api["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    # Convert the close_price, low_price, and high_price columns to float
    data_api["close_price"] = data_api["close_price"].astype(float)
    data_api["low_price"] = data_api["low_price"].astype(float)
    data_api["high_price"] = data_api["high_price"].astype(float)

    # Calculate the moving average
    moving_avg = float(data_api["close_price"].apply(float).mean())
    current_price = float(data_api.tail(1)["close_price"])
    cp = data_api["close_price"]
    lp = data_api["low_price"]
    hp = data_api["high_price"]
    print(data_api)
    # Calculate the Average True Range(ATR)
    # The value of the ATR is not directly related to the price of an asset. Instead, it reflects the degree of price fluctuation over a given time period. 
    # A higher ATR value indicates that the asset has had a larger range of price movements over the given time period, while a lower ATR value indicates a smaller range of price movements.
    data_api['ATR'] = talib.ATR(hp, lp, cp, timeperiod=24)
    # Calculate the rolling mean of ATR
    data_api['ATR_MA_4'] = data_api['ATR'].rolling(4).mean()
    # Flag the minutes where ATR breaks out its rolling mean
    data_api['ATR_breakout'] = np.where((data_api['ATR'] > data_api['ATR_MA_4']), True, False)
    # Calculate the three-candle rolling High
    data_api['three_candle_High'] = hp.rolling(3).max()
    # Check if the fourth candle is Higher than the Highest of the previous 3 candle
    data_api['four_candle_High'] = np.where( hp >
        data_api['three_candle_High'].shift(1), True, False)
    # Calculate the three-candle rolling Low
    data_api['three_candle_Low'] = lp.rolling(3).min()
    # Check if the fourth candle is Lower than the Lowest of the previous 3 candles
    data_api['four_candle_Low'] = np.where( lp <
        data_api['three_candle_Low'].shift(1), True, False)
    # Flag long positions
    data_api['long_positions'] = np.where(data_api['ATR_breakout'] & data_api['four_candle_High'], 1, 0)
    # Flag short positions
    data_api['short_positions'] = np.where(data_api['ATR_breakout'] & data_api['four_candle_Low'], -1, 0)
    # Combine long and short  position flags
    data_api['positions'] = data_api['long_positions'] + data_api['short_positions']
    
     # Check if the current price is higher or lower than the moving average
    if current_price > moving_avg:
        trend = "up"
    elif current_price < moving_avg:
        trend = "down"
    else:
        trend = "neutral"

    oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
    if oldest_data_from_stream_buffer:
        if skip_first_message:
            skip_first_message = False
            continue
        # Process the message and get the dataframe
        df_new = process_message(oldest_data_from_stream_buffer)
        # Append the new data to the dataframe
        df = df.append(df_new, ignore_index=True)
        # If the dataframe has more than 60 rows, drop the oldest row
        if len(df) > 60:
            df = df.iloc[1:]

        # If the dataframe has 60 rows, calculate and display the acceleration
        if len(df) == 60:
            # Calculate the speeds per second from the data in the dataframe
            speeds = calculate_speeds(df)

        # Calculate the average acceleration using the speeds and time intervals
            time_intervals = [1] * len(speeds)
            average_acceleration = calculate_average_acceleration(speeds, time_intervals)

            # # Display the average acceleration
            # if average_acceleration >= 0:
            #     print(termcolor.colored(f"Average acceleration: {average_acceleration:.2f}", 'green'))
            # else:
            #     print(termcolor.colored(f"Average acceleration: {average_acceleration:.2f}", 'red'))
        # If the average acceleration is positive and the position is currently not long, open a long position
            if average_acceleration >= 0 and position == 0 and trend == 'up':
                # Buy the asset if the conditions are met
                buy_price = current_price
                position = portfolio_value / buy_price
                portfolio_value += portfolio_value - (buy_price * position)
                SL = (buy_price * risk)
                with open("log.txt", "a") as f:
                    f.write("Bought: "  + str(position) + " @ " + str(buy_price) + "\n")
            # If the average acceleration is negative and the position is currently not short, open a short position
            elif average_acceleration < 0 and position > 0 and trend == 'down':
                # Sell the asset if the conditions are met
                sell_price = current_price
                
                # Calculate the profit/loss from the trade
                profit = sell_price - buy_price
                with open("log.txt", "a") as f:
                    f.write("Sold: "  + str(position) + " @ " + str(sell_price) + "PNL : " + str(profit) + "\n")
                # Update the portfolio value
                portfolio_value += profit
                position = 0
                with open("log.txt", "a") as f:
                    f.write("Portfolio Value: "  + str(portfolio_value) + "\n")
            else:
                pass
