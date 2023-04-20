import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import websocket
import json
import unicorn_binance_websocket_api
import pandas as pd
import datetime
import matplotlib.dates as mdates

data = []
ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1s'], ['btcusdt'])
skip_first_message = True

def process_message(msg):
    """Processes a websocket message and returns the close price."""
    data = json.loads(msg)
    k_dict = data['data']['k']
    df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
    close_time = pd.to_datetime(df['t'], unit='ms')
    close_price = df['c'].astype(float)
    return close_time, close_price

def update(num):
    skip_first_message = True
    while True:
        oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
        if oldest_data_from_stream_buffer:
            if skip_first_message:
                skip_first_message = False
                continue
            close_time, close_price = process_message(oldest_data_from_stream_buffer)
            data.append((close_time, close_price))
            plt.cla()
            times, prices = zip(*data)
            plt.plot(times, prices)
            
            # Get the start of the current hour
            start_of_hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
            
            # Set the x axis limits to cover the range from the start of the current hour to the current time
            ax.set_xlim(left=start_of_hour, right=datetime.datetime.now())
            
            # Set the x axis tick labels to display the time in the format "HH
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            # No data is available, so exit the loop
            break

fig, ax = plt.subplots()
ax.set_ylim([0, 100])
ani = animation.FuncAnimation(fig, update, interval=1000)
plt.show()
