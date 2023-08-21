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


# Define a function to process incoming websocket messages
def process_message(msg):
    data = json.loads(msg)
    k_dict = data['data']['k']

    try:
        df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
        return df

    except ValueError:
        print("Error: The shape of the data is different from the dataframe.")

ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
ubwa.create_stream(['kline_1m'], ['btcusdt'])
#channels = ['kline_5m', 'kline_15m', 'kline_30m', 'kline_1h', 'kline_12h', 'depth5']

skip_first_message = True
while True:
    oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
    if oldest_data_from_stream_buffer:
        if skip_first_message:
            skip_first_message = False
            continue
        process_message(oldest_data_from_stream_buffer)

