import unicorn_binance_websocket_api
import websocket
import json
import pandas as pd
import datetime
import numpy as np 
import termcolor
import os
import requests
import ATR
import pygame

if not os.path.exists('log.txt'):
  # Create the file if it does not exist
  open('log.txt', 'w').close()

last_price = 0
sum = 0
avg_speed_min = 0
avg_speed_5min = 0
avg_speed_15min = 0
long_position = 0
short_position = 0
risk_short = 1.01
greed_short = 0.98
risk_long = 0.98
greed_long = 1.01
portfolio_value = 10000
SLL = 0
TPL = 0
SLS = 0
TPS = 0
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

def log(message):
    with open("log.txt", "a") as f:
        f.write(message + '\n')

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
            log_message = 'Price: {} {:.2f}$/10sec --- {:.2f}$/min --- {:.2f}$/5min --- TOTAL: {:.2f}$'.format(last_price, cur_speed, avg_speed_min, avg_speed_5min, sum)
            log(log_message)

        # Buying long position
        if ATR.generate_trend() == 'uptrend' and avg_speed_5min > 0 and long_position == 0:
            # Buy the asset if the conditions are met
            buy_price = df.iloc[-1]['close_price']
            long_position = portfolio_value / buy_price
            portfolio_value += portfolio_value - (buy_price * long_position)
            SLL = (buy_price * risk_long)
            TPL = (buy_price * greed_long)
            with open("log.txt", "a") as f:
                f.write("Bought long: "  + str(long_position) + " @ " + str(buy_price) + '\n')
                f.write("TPL & SLL: "  + str(TPL) + " / " + str(SLL) + '\n')

        # Buying short position
        if ATR.generate_trend() == 'downtrend' and avg_speed_5min < 0 and short_position == 0:
            # Buy the asset if the conditions are met
            buy_price = df.iloc[-1]['close_price']
            short_position = portfolio_value / buy_price
            portfolio_value += portfolio_value - (buy_price * short_position)
            SLS = (buy_price * risk_short)
            TPS = (buy_price * greed_short)
            with open("log.txt", "a") as f:
                f.write("Bought short: "  + str(short_position) + " @ " + str(buy_price) + '\n')
                f.write("TPS & SLS: "  + str(TPS) + " / " + str(SLS) + '\n')

        # Selling Long position
        if (last_price <= SLL or last_price >= TPL) and (TPL !=0 and long_position != 0):
            with open("log.txt", "a") as f:
                f.write('close Long position @:' + str(last_price) + '\n')
            portfolio_value += portfolio_value - (last_price * long_position)
            long_position = 0 
            with open("log.txt", "a") as f:
                f.write('Portfolio :' + str(portfolio_value) + '\n')
        # Selling Short position
        if (last_price >= SLS or last_price <= TPS) and (TPS !=0 and short_position != 0):
            with open("log.txt", "a") as f:
                f.write('close Short position @:' + str(last_price) + '\n')
            portfolio_value += portfolio_value - (last_price * short_position)
            short_position = 0 
            with open("log.txt", "a") as f:
                f.write('Portfolio :' + str(portfolio_value) + '\n')
        





                # pygame.mixer.init()
                # pygame.mixer.music.load('go.mp3')
                # pygame.mixer.music.play()
                # pygame.time.delay(5000)
                # pygame.mixer.music.stop()






