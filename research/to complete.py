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
# Create an empty dataframe to store the close prices
df = pd.DataFrame(columns=['close_time', 'close_price'])

skip_first_message = True
# Connect to the websocket API and create a stream
ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1m'], ['btcusdt'])

def getHistorical():
    # Initialize an empty dataframe
    df = pd.DataFrame()
    # Set a flag to skip the first message
    skip_first_message = True
    
    # Keep getting data from the websocket indefinitely
    while True:
        # Get the next message from the websocket
        msg = ubwa.pop_stream_data_from_stream_buffer()
        
        # Skip the first message
        if skip_first_message:
            skip_first_message = False
            continue
        
        # If the dataframe has more than 15 rows, shift the rows by one index and drop the first row
        if len(df) > 15:
            df = df.shift(-1).drop(df.index[0])
        
        # Extract candle data from the message and create a new row in the dataframe
        k_dict = msg['k']
        candle_df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
        candle_df['close_timemous'] = pd.to_datetime(candle_df['t'], unit='ms')
        candle_df['close_price'] = candle_df['c'].astype(float)
        candle_df['low_price'] = candle_df['l'].astype(float)
        candle_df['high_price'] = candle_df['h'].astype(float)
        
        # Append the new row to the dataframe
        df = df.append(candle_df, ignore_index=True)
        return df


def process_message(msg):
    
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

while True:
    
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
        data = getHistorical()
        print(data)
        data['ATR'] = talib.ATR(hp, lp, cp, timeperiod=24)
        # Calculate the rolling mean of ATR
        data['ATR_MA_4'] = data['ATR'].rolling(4).mean()
        # Flag the minutes where ATR breaks out its rolling mean
        data['ATR_breakout'] = np.where((data['ATR'] > data['ATR_MA_4']), True, False)
        # Calculate the three-candle rolling High
        data['three_candle_High'] = hp.rolling(3).max()
        # Check if the fourth candle is Higher than the Highest of the previous 3 candle
        data['four_candle_High'] = np.where( hp >
            data['three_candle_High'].shift(1), True, False)
        # Calculate the three-candle rolling Low
        data['three_candle_Low'] = lp.rolling(3).min()
        # Check if the fourth candle is Lower than the Lowest of the previous 3 candles
        data['four_candle_Low'] = np.where( lp <
            data['three_candle_Low'].shift(1), True, False)
        # Flag long positions
        data['long_positions'] = np.where(data['ATR_breakout'] & data['four_candle_High'], 1, 0)
        # Flag short positions
        data['short_positions'] = np.where(data['ATR_breakout'] & data['four_candle_Low'], -1, 0)
        # Combine long and short  position flags
        data['positions'] = data['long_positions'] + data['short_positions']
        print(data)


        # Check if the current price is higher or lower than the moving average
        if current_price > moving_avg:
            trend = "uptrend"
        elif current_price < moving_avg:
            trend = "downtrend"
        else:
            trend = "neutral"
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
                    f.write("Bought: "  + str(position) + " @ " + str(buy_price))
                print('long')
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

                print('short')
            else:
                pass
