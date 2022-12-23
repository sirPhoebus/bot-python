import unicorn_binance_websocket_api
import websocket
import json
import pandas as pd
import datetime
import talib 
import numpy as np 
import termcolor


# "t": 123400000, // Kline start time
# "T": 123460000, // Kline close time
# "s": "BNBBTC",  // Symbol
# "i": "1m",      // Interval
# "f": 100,       // First trade ID
# "L": 200,       // Last trade ID
# "o": "0.0010",  // Open price
# "c": "0.0020",  // Close price
# "h": "0.0025",  // High price
# "l": "0.0015",  // Low price
# "v": "1000",    // Base asset volume
# "n": 100,       // Number of trades
# "x": false,     // Is this kline closed?
# "q": "1.0000",  // Quote asset volume
# "V": "500",     // Taker buy base asset volume
# "Q": "0.500",   // Taker buy quote asset volume
# "B": "123456"   // Ignore

# Initialize an empty list to store the previous positive speeds
previous_speeds = []

# Set the number of previous speeds to consider
num_prev_speeds = 10


# Define a function to process incoming websocket messages
def process_message(msg):
    data = json.loads(msg)
    # Extract the 'k' dictionary from the data
    k_dict = data['data']['k']
    #print(k_dict)
    # Convert the 'k' dictionary into a Pandas dataframe
    # Specify the column names
    #column_names = ["Start Time", "Close Time", "Symbol", "Interval", "First ID", "Last ID", "Open Price", "Close Price", "High Price", "Low Price", "Base Asset Volume", "Number of Trades", "Is this Kline Closed?", "Quote Asset Volume", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]

    try:
        # Convert the 'k' dictionary into a Pandas dataframe
        df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
        # In the main loop, after calling the function to get the current positive speed
        positive_speed = get_positive_speed(k_dict, previous_speeds)
        # Append the current positive speed to the list of previous speeds
        previous_speeds.append(positive_speed)
        # If the number of previous speeds exceeds the maximum number to consider
        if len(previous_speeds) > num_prev_speeds:
            # Remove the oldest positive speed from the list
            previous_speeds.pop(0)

    except ValueError:
        # Handle the error
        print("Error: The shape of the data is different from the dataframe.")
        # You can add any additional code here to handle the error, such as modifying the data or dataframe, or raising a different exception.

ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1m'], ['btcusdt'])
#channels = ['kline_5m', 'kline_15m', 'kline_30m', 'kline_1h', 'kline_12h', 'depth5']

def get_positive_speed(data, previous_speeds):
    try:
        # Convert the 'k' dictionary into a Pandas dataframe
        df = pd.DataFrame.from_dict(data, orient='index').transpose()
        # Calculate the duration of the candle
        duration = float(df['T']) - float(df['t'])
        # If the duration is zero, return a default value
        if duration == 0:
            return 0
        # Calculate the positive speed by dividing the difference between the high and low prices by the duration of the candle
        positive_speed = (float(df['h']) - float(df['l'])) / duration

        if len(previous_speeds) == 0:
            avg_prev_speeds = 0
        else:
            avg_prev_speeds = sum(previous_speeds) / len(previous_speeds)
        factor = 1000000
        formatted_speed = 'speed is: {:.2f} milliseconds'.format(positive_speed * factor)
        formatted_avg_prev_speeds = 'the average prev speeds is: {:.2f}'.format(avg_prev_speeds * factor)
        if positive_speed > avg_prev_speeds:
            # Output the strings in green
            print(termcolor.colored(formatted_speed, 'green'))
            print(termcolor.colored(formatted_avg_prev_speeds, 'green'))
            return positive_speed
        # If the positive speed is not above the average of the previous speeds
        elif positive_speed < avg_prev_speeds:
            # Output the strings in red
            print(termcolor.colored(formatted_speed, 'red'))
            print(termcolor.colored(formatted_avg_prev_speeds, 'red'))
            # Return 0
            return 0
        else:
            # Return the positive speed
            return 0
    # If an error occurs while processing the data
    except ValueError:
        # Handle the error
        print("Error: The shape of the data is different from the dataframe.")
        # You can add any additional code here to handle the error, such as modifying the data or dataframe, or raising a different exception.


skip_first_message = True
while True:
    oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
    if oldest_data_from_stream_buffer:
        if skip_first_message:
            skip_first_message = False
            continue
        process_message(oldest_data_from_stream_buffer)
