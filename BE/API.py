import pandas as pd
import datetime
import numpy as np 
import requests
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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

def getAggVolume():
    url= "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=2&page=1&sparkline=false&price_change_percentage=1h%2C24h"
    response = requests.get(url).json()
    return response

def getGlobalData():
    url= "https://api.coingecko.com/api/v3/global"
    response = requests.get(url).json()
    return response

def calculate_ATR(symbol:str, tf:str):
    df = pd.DataFrame(getHistorical(symbol,tf), columns=[
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
    df["open_time"] = df["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    # Convert the close_price, low_price, and high_price columns to float
    df["close_price"] = df["close_price"].astype(float)
    df["low_price"] = df["low_price"].astype(float)
    df["high_price"] = df["high_price"].astype(float)

    # Calculate the moving average
    moving_avg = float(df["close_price"].apply(float).mean())
    #current_price = float(df.tail(1)["close_price"])
    current_price = df.iloc[-1]["close_price"]
    cp = df["close_price"]
    lp = df["low_price"]
    hp = df["high_price"]

    # Calculate the Average True Range(ATR)
    # The value of the ATR is not directly related to the price of an asset. Instead, it reflects the degree of price fluctuation over a given time period. 
    # A higher ATR value indicates that the asset has had a larger range of price movements over the given time period, while a lower ATR value indicates a smaller range of price movements.

    df['ATR'] = pd.DataFrame({'hp': hp, 'lp': lp, 'cp': cp}).apply(lambda x: x.max() - x.min(), axis=1)
    # Calculate the rolling mean of ATR using a 4-period window
    df['ATR_MA_4'] = df['ATR'].ewm(span=4).mean()
    # Flag the minutes where ATR breaks out its rolling mean
    df['ATR_breakout'] = np.where((df['ATR'] > df['ATR_MA_4']), True, False)
    # Calculate the three-candle rolling High
    df['three_candle_High'] = hp.rolling(3).max()
    # Check if the fourth candle is Higher than the Highest of the previous 3 candle
    df['four_candle_High'] = np.where(hp > df['three_candle_High'].shift(1), True, False)
    # Calculate the three-candle rolling Low
    df['three_candle_Low'] = lp.rolling(3).min()
    # Check if the fourth candle is Lower than the Lowest of the previous 3 candles
    df['four_candle_Low'] = np.where(lp < df['three_candle_Low'].shift(1), True, False)
    # Flag long positions
    df['long_positions'] = np.where(df['ATR_breakout'] & df['four_candle_High'], 1, 0)
    # Flag short positions
    df['short_positions'] = np.where(df['ATR_breakout'] & df['four_candle_Low'], -1, 0)
    # Combine long and short position flags
    df['positions'] = df['long_positions'] + df['short_positions']
    # Check if the current price is higher or lower than the moving average
    if current_price > moving_avg:
        trend = "uptrend"
    elif current_price < moving_avg:
        trend = "downtrend"
    else:
        trend = "neutral"
    return trend

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

def getOrderBook(symbol):
    r = requests.get("https://api.binance.com/api/v3/depth",
                 params=dict(symbol=symbol, limit="5000"))
    results = r.json()
    frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],dtype=float) for side in ["bids", "asks"]}
    frames_list = [frames[side].assign(side=side) for side in frames]
    data = pd.concat(frames_list, axis="index", ignore_index=True, sort=True)
    price_summary = data.groupby("side").price.describe()
    price_summary.to_markdown()
    return price_summary

def calculate_BB_trend(symbol, tf):
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
        return "Above"
    else:
        return "Below"

def getRussianDollTrend(symbol):
    russianDoll = []
    trend, bullish_count = getTrendTimeframe(symbol, '1h')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe(symbol, '30m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe(symbol, '15m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe(symbol, '5m')
    russianDoll += [bullish_count - 250]
    trend, bullish_count = getTrendTimeframe(symbol, '1m')
    russianDoll += [bullish_count - 250]
    return russianDoll

async def run_getTrendTimeframe(symbol:str, tf: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, getTrendTimeframe, symbol, tf)


def getTrendTimeframe (symbol, tf):
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

app = FastAPI()
# Set up CORS
origins = [
    "http://localhost:4200",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"Working..."}

@app.get("/white_soldiers")
async def get_data(symbol:str, tf: str):
    task = asyncio.create_task(run_getTrendTimeframe(symbol, tf))
    data = await task
    return {"data": data}

@app.get("/russian_doll")
async def get_data(symbol:str):
    data = getRussianDollTrend(symbol)
    return {"data": data}

@app.get("/bb_trend")
async def get_data(symbol: str, tf: str):
    data = calculate_BB_trend(symbol, tf)
    return {"data": data}

@app.get("/vwap")
async def get_data(symbol: str, tf: str):
    data = calculate_vwap(symbol, tf)
    return {"data": data}

@app.get("/atr")
async def get_data(symbol: str, tf: str):
    data = calculate_ATR(symbol, tf)
    return {"data": data}

@app.get("/ob")
async def get_data(symbol: str):
    data = getOrderBook(symbol)
    return {"data": data}

@app.get("/agg_vol")
async def get_data():
    data = getAggVolume()
    return {"data": data}

@app.get("/global")
def get_data():
    data = getGlobalData()
    return {"data": data}

#launch fastapi server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
