import unicorn_binance_websocket_api
import json
import pandas as pd
import datetime
import numpy as np 
import time
import requests

#define sybol
symbol = 'BTCUSDT'

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
mult = 2
length = 20
nbrOfBullishCandles = 3

# Define timeframes
tf_higher = '1d' # Higher time-frame
tf_lower = '1h' # Lower time-frame

def getHistorical(symbol, tf):
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    # "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"
    params = {
        'symbol': symbol,
        'interval': tf 
    }

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check the status code to make sure the request was successful
    # Convert the response to a JSON object
    df = response.json()    
    return df
def calculate_BB_trend(df):
    df["close_price"] = df["close_price"].astype(float)
    df["low_price"] = df["low_price"].astype(float)
    df["high_price"] = df["high_price"].astype(float)

    current_price = df.iloc[-1]["close_price"]
    hp = df["high_price"]
    basis = np.mean(hp[-length:])
    dev = mult * np.std(hp[-length:])
    upper = basis + dev
    lower = basis - dev
    if current_price <= lower:
        #Lower Bound Hit!
        trend = 'BUY'
    if current_price >= upper:
        trend = 'SELL'
    if current_price > lower or current_price < upper:
        trend = 'In range'
    return trend
def calculate_vwap(symbol, tf):
    df = pd.DataFrame(getHistorical(symbol, tf), columns=[
    "open_time",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "unused_field"
    ])
    df["close_price"] = df["close_price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    # Create a new column for the cumulative volume
    df['cum_volume'] = df['volume'].cumsum()
    df["cum_volume"] = df["cum_volume"].astype(float)
    df['VWAP'] = (df['cum_volume'] - df['volume'].rolling(period).sum() ) / (df['cum_volume'] - df['cum_volume'].rolling(period).sum() ) * df['close_price']
    # Calculate the short and long period moving averages
    df['short_ma'] = df['VWAP'].rolling(window=5).mean()
    df['long_ma'] = df['VWAP'].rolling(window=20).mean()
    if df['short_ma'].iloc[-1] > df['long_ma'].iloc[-1]:
        return "BUY"
    else:
        return "SELL"
def getRussianDollTrend():
    russianDoll = []
    trend, bullish_count = getTrendTimeframe('1h')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe('30m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe('15m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe('5m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe('1m')
    russianDoll += [bullish_count - 250]
    return str(russianDoll)
def getTrendTimeframe (tf):
    # Get historical data
    dt = getHistorical(symbol, tf)

    # Extract the relevant data from the historical candles
    open_lower = np.array([candle[1] for candle in dt])
    close_lower = np.array([candle[4] for candle in dt])

    # Define a function to determine if a candle is bullish or bearish
    def is_bullish_candle(open, close):
        return close > open

    # Calculate the number of bullish candles on the lower time-frame
    bullish_count = 0
    for i in range(len(open_lower)):
        if is_bullish_candle(open_lower[i], close_lower[i]):
            bullish_count += 1

    # Determine if the majority of candles on the lower time-frame are bullish
    trend = (bullish_count > len(open_lower) / 2)
    return trend, bullish_count

count = 0
while True:
    count += 1
    if count % (3*60) == 0:
        s = getRussianDollTrend()
        arr = np.array(eval(s))  # Use eval to convert string to list, then convert to NumPy array
        b_bulls = np.sum(arr) > nbrOfBullishCandles
        s_bulls = np.sum(arr) < (nbrOfBullishCandles * -1)
        vwap  = calculate_vwap(symbol, '1h')
        bb = calculate_BB_trend(symbol, '5m')
        #Get BTC price
        response = requests.get("https://api.coinbase.com/v2/prices/spot?currency=USD")
        # parse the response to get the current BTC price
        price = response.json()["data"]["amount"]
        if bb == 'BUY' and b_bulls and vwap == 'BUY' and long_position == 0 and short_position == 0:
            long_position = portfolio_value / price
            portfolio_value += portfolio_value - (price * long_position)
            SLL = (price * risk_long)
            TPL = (price * greed_long)
            print("Bought long: "  + str(long_position) + " @ " + str(price) + '\n')
            print("TPL & SLL: "  + str(TPL) + " / " + str(SLL) + '\n')
        elif bb == 'SELL' and s_bulls and vwap == 'SELL' and long_position == 0 and short_position == 0:
            short_position = portfolio_value / price
            portfolio_value += portfolio_value - (price * short_position)
            SLS = (price * risk_short)
            TPS = (price * greed_short)
            print("Bought short: "  + str(short_position) + " @ " + str(price) + '\n')
            print("TPS & SLS: "  + str(TPS) + " / " + str(SLS) + '\n')

        elif (price <= SLL or price >= TPL) and TPL !=0 and long_position != 0:
            print('close Long position @:' + str(price) + '\n')
            portfolio_value = price * long_position
            long_position = 0
            SLL = 0
            TPL = 0
            print('Portfolio :' + str(portfolio_value) + '\n')
        # Selling Short position
        elif (price >= SLS or price <= TPS) and (TPS !=0 and short_position != 0):
            print('close Short position @:' + str(price) + '\n')
            portfolio_value = price * short_position
            short_position = 0 
            SLS = 0 
            TPS = 0
            print('Portfolio :' + str(portfolio_value) + '\n')
    time.sleep(1)  # Wait 1 second before looping again


