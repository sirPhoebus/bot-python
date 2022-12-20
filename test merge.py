import json
import logging
import hmac
import time
import hashlib
import requests
from urllib.parse import urlencode
import pandas as pd
import json
import time
import talib
from math import *

#KEY = 'a2db807694b8361004d1f09f27bed70154659c1cc3d5aba883692e064551fbd0'
#SECRET = '5e127543b9e4545e439daf4e19879a09954c3547a088acd781c8c72c9683ec21'
KEY = 'M0j9VyCdXXtkOxwrubZaBHeqjK3s0TxyDambJ7ZkQzveeeu1DCzLxG8q9kvc5To3'
SECRET = 'wh8cQ8WMzbTv170B6QknVOnjL0X9nAmgVZWPcQElCqDBBIHpWhWCCNZ0UdiAouqM'


# testnet base url : https://testnet.binancefuture.com
# prod URL for futures : https://fapi.binance.com
BASE_URL = "https://api1.binance.com"  
SECRET_KEY = "whatthefuckkey"
side = ""

def hashing(query_string):
    return hmac.new(
        SECRET.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def get_timestamp():
    return int(time.time() * 1000)


def dispatch_request(http_method):
    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json;charset=utf-8", "X-MBX-APIKEY": KEY}
    )
    return {
        "GET": session.get,
        "DELETE": session.delete,
        "PUT": session.put,
        "POST": session.post,
    }.get(http_method, "GET")


# used for sending request requires the signature
def send_signed_request(http_method, url_path, payload={}):
    query_string = urlencode(payload)
    # replace single quote to double quote
    query_string = query_string.replace("%27", "%22")
    if query_string:
        query_string = "{}&timestamp={}".format(query_string, get_timestamp())
    else:
        query_string = "timestamp={}".format(get_timestamp())

    url = (
            BASE_URL + url_path + "?" + query_string + "&signature=" + hashing(query_string)
    )
    print("{} {}".format(http_method, url))
    params = {"url": url, "params": {}}
    response = dispatch_request(http_method)(**params)
    print('send_signed_request' + str(response.json()))
    return response.json()


# used for sending public data request
def send_public_request(url_path, payload={}):
    query_string = urlencode(payload, True)
    url = BASE_URL + url_path
    if query_string:
        url = url + "?" + query_string
    print("{}".format(url))
    response = dispatch_request("GET")(url=url)
    #print('send_public_request' + str(response.json()))
    return response.json()


# CONSTANTS
pairSymbol = 'ETHUSDT'
fiatSymbol = 'USDT'
cryptoSymbol = 'ETH'
trixLength = 9
trixSignal = 21

# API
binance_api_key = ''  # Enter your own API-key here
binance_api_secret = ''  # Enter your own API-secret here


def getHistorical(symbole):
# Set the API endpoint URL
    url = 'https://api.binance.com/api/v3/klines'

    # Set the request parameters
    params = {
        'symbol': 'BNBBTC',
        'interval': '1h'
    }

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check the status code to make sure the request was successful
    # Convert the response to a JSON object
    data = response.json()
    json_data = json.dumps(data)
    df = pd.read_json(json_data)
    # Normalize the JSON response and put it into a Pandas dataframe
    #df = pd.json_normalize(data)
    #print(df)
    # You can now access the data in the dataframe using the column names

    return df

def getBalance(myclient, coin):
    jsonBalance = myclient.get_balances()
    if jsonBalance == []:
        return 0
    pandaBalance = pd.DataFrame(jsonBalance)
    if pandaBalance.loc[pandaBalance['coin'] == coin].empty:
        return 0
    else:
        return float(pandaBalance.loc[pandaBalance['coin'] == coin]['free'])

def get_step_size(symbol):
    # stepSize = None
    # for filter in client.get_symbol_info(symbol)['filters']:
    #     if filter['filterType'] == 'LOT_SIZE':
    #         stepSize = float(filter['stepSize'])
    return 2

def get_price_step(symbol):
    # stepSize = None
    # for filter in client.get_symbol_info(symbol)['filters']:
    #     if filter['filterType'] == 'PRICE_FILTER':
    #         stepSize = float(filter['tickSize'])
    return 2

def convert_amount_to_precision(symbol, amount):
    stepSize = get_step_size(symbol)
    return (amount//stepSize)*stepSize

def convert_price_to_precision(symbol, price):
    stepSize = get_price_step(symbol)
    return (price//stepSize)*stepSize

df = getHistorical(pairSymbol)

df['TRIX'] = talib.trend.ema_indicator(talib.trend.ema_indicator(talib.trend.ema_indicator(close=df['close'], window=trixLength), window=trixLength), window=trixLength)
df['TRIX_PCT'] = df["TRIX"].pct_change()*100
df['TRIX_SIGNAL'] = talib.trend.sma_indicator(df['TRIX_PCT'],trixSignal)
df['TRIX_HISTO'] = df['TRIX_PCT'] - df['TRIX_SIGNAL']
df['STOCH_RSI'] = talib.momentum.stochrsi(close=df['close'], window=15, smooth1=3, smooth2=3)
print(df)

actualPrice = df['close'].iloc[-1]
#fiatAmount = float(client.get_asset_balance(asset=fiatSymbol)['free'])
fiatAmount = 10000
#cryptoAmount = float(client.get_asset_balance(asset=cryptoSymbol)['free'])
cryptoAmount = 1
minToken = 5/actualPrice
print('coin price :',actualPrice, 'usd balance', fiatAmount, 'coin balance :',cryptoAmount)

def buyCondition(row, previousRow):
    if row['TRIX_HISTO'] > 0 and row['STOCH_RSI'] <= 0.82:
        return True
    else:
        return False

def sellCondition(row, previousRow):
    if row['TRIX_HISTO'] < 0 and row['STOCH_RSI'] >= 0.2:
        return True
    else:
        return False

if buyCondition(df.iloc[-2], df.iloc[-3]):
    if float(fiatAmount) > 5:
        quantityBuy = convert_amount_to_precision(pairSymbol, 0.98 * (float(fiatAmount)/actualPrice))
        # buyOrder = client.order_market_buy(
        #     symbol=pairSymbol,
        #     quantity=quantityBuy)
        #send_signed_request("POST", "/sapi/v1/margin/order", params)
        print("BUY: ", pairSymbol, quantityBuy)
    else:
        pass
        print("If you  give me more USD I will buy more",cryptoSymbol) 

elif sellCondition(df.iloc[-2], df.iloc[-3]):
    if float(cryptoAmount) > minToken:
        # sellOrder = client.order_market_sell(
        #     symbol=pairSymbol,
        #     quantity=convert_amount_to_precision(pairSymbol, cryptoAmount))
        print("BUY: ", pairSymbol, convert_amount_to_precision(pairSymbol, cryptoAmount))
    else:
        pass
        print("If you give me more",cryptoSymbol,"I will sell it")
else :
    pass
    print("No opportunity to take")