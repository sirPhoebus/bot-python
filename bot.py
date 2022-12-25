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
import ATR

sum = 0
avg_speed_min = 0
avg_speed_15min = 0

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
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%H:%M:%S")
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
        # If the dataframe has more than 900 rows, drop the oldest row
        if len(df) > 900:
            df = df.iloc[1:]
        speeds = calculate_speeds(df)
        if len(df) == 900:
            avg_speed_15min = np.mean(speeds[-900])
            #print('Avg ($/15min): {:.2f}'.format(avg_speed_15min))
        if len(df) > 60:
            avg_speed_min = np.mean(speeds[-60:])
            #print('Avg ($/min): {:.2f}'.format(avg_speed_min))
        if len(df) > 3:
            cur_speed = np.mean(speeds[-3:])
            sum = sum + cur_speed 
            
            
            #print the last price from the dataframe
            print("Bot started at: " + str(formatted_time) + " --- Last Close Price: {}".format(df.iloc[-1]['close_price']))
            print('Last candle: {:.2f}'.format(cur_speed) + '$' + ' --- {:.2f}'.format(avg_speed_min) + '$/min) --- {:.2f}'.format(avg_speed_15min) + '($/15min) --- TOTAL: {:.2f}'.format(sum) + '$')
            print(ATR.generate_trend())




