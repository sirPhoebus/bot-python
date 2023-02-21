from flask import Flask, render_template
import unicorn_binance_websocket_api
import json
import pandas as pd
import datetime
import numpy as np
import requests
from plotly.offline import plot
import plotly.graph_objs as go
import ATR


app=Flask(__name__,template_folder='templates')
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
 

@app.route("/")
def index():
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
            dff = calculate_vwap(df)
            # Create the plot
            plot_data = [go.Scatter(x=dff['close_time'], y=dff['VWAP'])]

            plot_layout = go.Layout(title='VWAP')
            fig = go.Figure(data=plot_data, layout=plot_layout)
            plot_div = plot(fig, output_type='div')
        
            return render_template("index.html", plot_div=plot_div)


if __name__ == "__main__":
    app.run(debug=True)
