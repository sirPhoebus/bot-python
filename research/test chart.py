import unicorn_binance_websocket_api
import websocket
import json
import pandas as pd
import datetime
import talib 
import numpy as np 
import matplotlib.pyplot as plt
import asyncio

def process_message(msg):
    """Processes a websocket message and adds the close price to the dataframe."""
    data = json.loads(msg)
    k_dict = data['data']['k']
    df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
    df['close_time'] = pd.to_datetime(df['t'], unit='ms')
    df['close_price'] = df['c'].astype(float)
    df = df[['close_time', 'close_price']]
    return df

# Connect to the websocket API and create a stream
async def main():
    ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
    ubwa.create_stream(['kline_1m'], ['btcusdt'])

    # Create an empty dataframe to store the close prices
    df = pd.DataFrame(columns=['close_time', 'close_price'])

    skip_first_message = True
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    mult = 2
    length = 20
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

            # Calculate the SMA and upper and lower bands using TA-Lib
            df["basis"] = talib.SMA(df["close_price"], length)
            df["dev"] = mult * talib.STDDEV(df["close_price"], length)
            df["upper"] = df["basis"] + df["dev"]
            df["lower"] = df["basis"] - df["dev"]

            # Plot the basis, upper, lower, and moving average columns on the axis
            # (Assume that the data for these columns is stored in variables named 'basis', 'upper', 'lower', and 'moving_average', respectively)
            fig, ax = plt.subplots()
            ax.plot(df["basis"], label='Basis')
            ax.plot(df["upper"], label='Upper')
            ax.plot(df["lower"], label='Lower')
            #ax.axhline(y=df["close_price"], label='Close price', color='r')
            ax.legend()
            ax.plot()
            

while True:
    asyncio.run(main())



