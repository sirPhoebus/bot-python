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
import pygame

if not os.path.exists('log.txt'):
  # Create the file if it does not exist
  open('log.txt', 'w').close()

sum = 0
avg_speed_min = 0
avg_speed_5min = 0
avg_speed_15min = 0
position = 0
risk = 0.95
greed = 1.05
# Create a variable to keep track of the portfolio value
portfolio_value = 10000
SL = 0
TP = 0
gain = 0 

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
with open("log.txt", "a") as f:
        f.write("Bot started at: " + str(formatted_time) + '\n')
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
        if len(df) > 911:
            df = df.iloc[1:]
        speeds = calculate_speeds(df)
        if len(df) > 900:
            avg_speed_15min = np.mean(speeds[-900])
            #print('Avg ($/15min): {:.2f}'.format(avg_speed_15min))
        if len(df) > 300:
            avg_speed_5min = np.mean(speeds[-300:])
            #print('Avg ($/min): {:.2f}'.format(avg_speed_min))
        if len(df) > 60:
            avg_speed_min = np.mean(speeds[-60:])
            #print('Avg ($/min): {:.2f}'.format(avg_speed_min))
        if len(df) > 10:
            cur_speed = np.mean(speeds[-3:])
            sum = sum + cur_speed 
            
            last_price = float(df.iloc[-1]['close_price'])
            #print the last price from the dataframe
            
            with open("log.txt", "a") as f:
                    f.write('Price:' + str(last_price) + ' {:.2f}'.format(cur_speed) + '$/10sec' + ' --- {:.2f}'.format(avg_speed_min) + '$/min --- {:.2f}'.format(avg_speed_5min) + '$/5min --- TOTAL: {:.2f}'.format(sum) + '$' + '\n')

            if ATR.generate_trend() == 'uptrend' and avg_speed_5min > 0 and position == 0:
                # Buy the asset if the conditions are met
                buy_price = df.iloc[-1]['close_price']
                position = portfolio_value / buy_price
                portfolio_value += portfolio_value - (buy_price * position)
                SL = (buy_price * risk)
                TP = (buy_price * greed)
                with open("log.txt", "a") as f:
                    f.write("Bought: "  + str(position) + " @ " + str(buy_price) + '\n')

            if (last_price <= SL or last_price >= TP) and (TP !=0 and position != 0):
                with open("log.txt", "a") as f:
                    f.write('close position @:' + str(last_price) + '\n')
                portfolio_value += portfolio_value - (last_price * position)
                position = 0 
                with open("log.txt", "a") as f:
                    f.write('Portfolio :' + str(portfolio_value) + '\n')
                pygame.mixer.init()
                pygame.mixer.music.load('go.mp3')
                pygame.mixer.music.play()
                pygame.time.delay(5000)
                pygame.mixer.music.stop()






