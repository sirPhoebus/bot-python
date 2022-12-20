# Import the necessary libraries
import pandas as pd
import os
import requests

if not os.path.exists('log.txt'):
  # Create the file if it does not exist
  open('log.txt', 'w').close()
# Load the historical data for the asset
data = pd.read_csv('asset_data.csv')

# Calculate the 100-day and 200-day moving averages
ma_100 = data['Close'].rolling(window=100).mean()
ma_200 = data['Close'].rolling(window=200).mean()

# Calculate the relative strength index of the asset's price
rsi = data['Close'].rolling(window=20).apply(lambda x: 100 - (100 / (1 + (x.mean() / x.std()))))

# Calculate the Bollinger Bands for the asset's price
bb_mean = data['Close'].rolling(window=20).mean()
bb_std = data['Close'].rolling(window=20).std()
bb_upper = bb_mean + 2 * bb_std
bb_lower = bb_mean - 2 * bb_std

# Create a variable to keep track of the current position (0 = no position, 1 = long position, -1 = short position)
position = 0
risk = 0.95
# Create a variable to keep track of the portfolio value
portfolio_value = 10000
SL = 0
market_perf = portfolio_value / int(data['Close'].iloc[1])
# Loop through the data and make buy/sell decisions
for i in range(len(data)):
    # Check if the 100-day moving average is crossing over the 200-day moving average : ma_100.iloc[i] > ma_200.iloc[i]
    # check RSI : rsi.iloc[-1] < 30
    
    if  rsi.iloc[i] > 31 and position == 0:
        # Buy the asset if the conditions are met
        buy_price = data['Close'].iloc[i]
        position = portfolio_value / buy_price
        portfolio_value += portfolio_value - (buy_price * position)
        SL = (buy_price * risk)
        with open("log.txt", "a") as f:
            f.write("Bought: "  + str(position) + " @ " + str(buy_price) + " on : " + str(data['Date'].iloc[i]) + "\n")

    # todo
    elif data['Close'].iloc[i] <= SL or data['Close'].iloc[i] > bb_upper.iloc[i] and position > 0:
        # Sell the asset if the conditions are met
        sell_price = data['Close'].iloc[i]
        
        # Calculate the profit/loss from the trade
        profit = sell_price - buy_price
        with open("log.txt", "a") as f:
            f.write("Sold: "  + str(position) + " @ " + str(sell_price) + " on : " + str(data['Date'].iloc[i]) + "PNL : " + str(profit) + "\n")

        # Update the portfolio value
        portfolio_value += profit
        position = 0
        with open("log.txt", "a") as f:
            f.write("Portfolio Value: "  + str(portfolio_value) + "\n")
    else:
        pass

# make a GET request to the Coinbase API to retrieve the current price of BTC
response = requests.get("https://api.coinbase.com/v2/prices/spot?currency=USD")

# parse the response to get the current BTC price
btc_price = response.json()["data"]["amount"]
with open("log.txt", "a") as f:
            f.write("MARKET PERF: "  + str(int(market_perf) * btc_price) + "\n")
