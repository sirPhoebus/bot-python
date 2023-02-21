import unicorn_binance_websocket_api
import json
import time
import requests
import pandas as pd
import numpy as np 
import datetime

def getOrderBook():
    r = requests.get("https://api.binance.com/api/v3/depth",
                 params=dict(symbol="BTCUSDT", limit="5000"))
    results = r.json()
    frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],dtype=float) for side in ["bids", "asks"]}
    frames_list = [frames[side].assign(side=side) for side in frames]
    data = pd.concat(frames_list, axis="index", ignore_index=True, sort=True)
    price_summary = data.groupby("side").price.describe()
    price_summary.to_markdown()
    return price_summary

def getHistoricalCandles():
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '5m',
        'limit': '288' # max 1000
    }

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check the status code to make sure the request was successful
    # Convert the response to a JSON object
    df = response.json()    
    return df

def processCanddles(msg):
    data = json.loads(msg)
    print(data)
    k_dict = data['data']['k']
    df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
    df['close_time'] = pd.to_datetime(df['t'], unit='ms')
    df['close_price'] = df['c'].astype(float)
    df['volume'] = df['v'].astype(float)
    df["low_price"] = df['l'].astype(float)
    df["high_price"] = df['h'].astype(float)
    df = df[['close_time', 'close_price','volume', 'low_price', 'high_price']]
    return df

def processOrderBook(msg):
    response = json.loads(msg)
    bids = pd.DataFrame(response['data']['bids'], columns=['Bids', 'Quantity'])
    asks = pd.DataFrame(response['data']['asks'], columns=['Asks', 'Quantity'])
    print(bids, '\n', asks)

print(getOrderBook())
# # Connect to the websocket API and create a stream
# ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
# #ubwa.create_stream(['kline_5m'], ['btcusdt'])
# ubwa.create_stream(['depth20'], ['ethusdt'])
# skip_first_message = True

# df = pd.DataFrame(getHistoricalCandles(), columns=[
#         "open_time",
#         "open_price",
#         "high_price",
#         "low_price",
#         "close_price",
#         "volume",
#         "close_time",
#         "quote_asset_volume",
#         "number_of_trades",
#         "taker_buy_base_asset_volume",
#         "taker_buy_quote_asset_volume",
#         "unused_field"
#         ])
# df["open_time"] = df["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
# # Convert the close_price, low_price, and high_price columns to float
# df["close_price"] = df["close_price"].astype(float)
# df["low_price"] = df["low_price"].astype(float)
# df["high_price"] = df["high_price"].astype(float)

# Start Websocket
# while True:
#     oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
#     if oldest_data_from_stream_buffer:
#         if skip_first_message:
#             skip_first_message = False
#             continue
#         # Process the message and get the dataframe
#         # df_new = processCanddles(oldest_data_from_stream_buffer)
#         processOrderBook(oldest_data_from_stream_buffer)
#         time.sleep(5)