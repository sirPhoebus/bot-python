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

last_price = 0
sum = 0
avg_speed_min = 0
avg_speed_5min = 0
avg_speed_15min = 0
long_position = 0
short_position = 0
risk_short = 1.005
greed_short = 0.995
risk_long = 0.995
greed_long = 1.005
portfolio_value = 10000
SLL = 0
TPL = 0
SLS = 0
TPS = 0
gain = 0 
period = 14

def process_message(msg):
    """Processes a websocket message and adds the close price to the dataframe."""
    data = json.loads(msg)
    k_dict = data['data']['k']
    df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
    df['close_time'] = pd.to_datetime(df['t'], unit='ms')
    df['close_price'] = df['c'].astype(float)
    df['volume'] = df['v'].astype(float)
    df = df[['close_time', 'close_price','volume']]
    return df

def calculate_speeds(df):
    """Calculates the speeds per second from the data in the dataframe."""
    # Extract the close prices from the dataframe
    close_prices = df['close_price'].values
    # Calculate the speeds as the difference between consecutive close prices
    speeds = np.diff(close_prices)
    # Return the speeds per second
    return speeds

def calculate_vwap(df):
    df["close_price"] = df["close_price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    # Create a new column for the cumulative volume
    df['cum_volume'] = df['volume'].cumsum()
    df["cum_volume"] = df["cum_volume"].astype(float)
    df['VWAP'] = (df['cum_volume'] - df['volume'].rolling(period).sum() ) / (df['cum_volume'] - df['cum_volume'].rolling(period).sum() ) * df['close_price']
    # Calculate the short and long period moving averages
    df['short_ma'] = df['VWAP'].rolling(window=5).mean()
    df['long_ma'] = df['VWAP'].rolling(window=20).mean()
    # Create a new column for the signal
    df['signal'] = None
    # Generate a buy signal when the short MA crosses above the long MA
    df.loc[(df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1)), 'signal'] = 'Buy'
    # Generate a sell signal when the short MA crosses below the long MA
    df.loc[(df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1)), 'signal'] = 'Sell'
    return df
    
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
            print(calculate_vwap(df).tail(1))
            #print the last price from the dataframe
            log_message = 'Price: {} {:.2f}$/10sec --- {:.2f}$/min --- {:.4f}$/5min --- TOTAL: {:.2f}$'.format(last_price, cur_speed, avg_speed_min, avg_speed_5min, sum)
            print(log_message)

        # Buying long position
        if ATR.generate_trend() == 'uptrend' and sum >= 5 and long_position == 0 and short_position == 0:
            # Buy the asset if the conditions are met
            buy_price = df.iloc[-1]['close_price']
            long_position = portfolio_value / buy_price
            portfolio_value += portfolio_value - (buy_price * long_position)
            SLL = (buy_price * risk_long)
            TPL = (buy_price * greed_long)
            print("Bought long: "  + str(long_position) + " @ " + str(buy_price) + '\n')
            print("TPL & SLL: "  + str(TPL) + " / " + str(SLL) + '\n')

        # Buying short position
        if ATR.generate_trend() == 'downtrend' and sum <= -5 and short_position == 0 and long_position == 0:
            # Buy the asset if the conditions are met
            buy_price = df.iloc[-1]['close_price']
            short_position = portfolio_value / buy_price
            portfolio_value += portfolio_value - (buy_price * short_position)
            SLS = (buy_price * risk_short)
            TPS = (buy_price * greed_short)
            print("Bought short: "  + str(short_position) + " @ " + str(buy_price) + '\n')
            print("TPS & SLS: "  + str(TPS) + " / " + str(SLS) + '\n')

        # Selling Long position
        if (last_price <= SLL or last_price >= TPL) and (TPL !=0 and long_position != 0):
            print('close Long position @:' + str(last_price) + '\n')
            portfolio_value += portfolio_value - (last_price * long_position)
            long_position = 0 
            print('Portfolio :' + str(portfolio_value) + '\n')
        # Selling Short position
        if (last_price >= SLS or last_price <= TPS) and (TPS !=0 and short_position != 0):
            print('close Short position @:' + str(last_price) + '\n')
            portfolio_value += portfolio_value - (last_price * short_position)
            short_position = 0 
            print('Portfolio :' + str(portfolio_value) + '\n')
        





