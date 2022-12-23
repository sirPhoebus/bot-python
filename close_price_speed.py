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

# Connect to the websocket API and create a stream
ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1m'], ['btcusdt'])

# Create an empty dataframe to store the close prices
df = pd.DataFrame(columns=['close_time', 'close_price'])

skip_first_message = True
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
            avg_speed = np.mean(speeds)

            if avg_speed >= 0:
                print(termcolor.colored('Avg ($/min): {:.2f}'.format(avg_speed)), 'green')
            else:
                print(termcolor.colored('Avg: {:.2f}'.format(avg_speed)), 'red')

            cur_speed = np.mean(speeds[-3:])
            if cur_speed >= 0:
                print(termcolor.colored('Current ($/sec): {:.2f}'.format(cur_speed)), 'green')
            else:
                print(termcolor.colored('Current: {:.2f}'.format(cur_speed)), 'red')



