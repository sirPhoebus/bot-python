# Import the necessary libraries
import pandas as pd
import numpy as np 
import talib 
from tqdm import tqdm

# Load the historical data for the asset
data = pd.read_csv('asset_data.csv')

# Calculate the Average True Range(ATR)
data['ATR'] = talib.ATR(data['High'], data['Low'],
                        data['Close'], timeperiod=24)
# Calculate the rolling mean of ATR
data['ATR_MA_4'] = data['ATR'].rolling(4).mean()
# Flag the minutes where ATR breaks out its rolling mean
data['ATR_breakout'] = np.where((data['ATR'] > data['ATR_MA_4']), True, False)
# Calculate the three-candle rolling High
data['three_candle_High'] = data['High'].rolling(3).max()
# Check if the fourth candle is Higher than the Highest of the previous 3 candle
data['four_candle_High'] = np.where( data['High'] >
    data['three_candle_High'].shift(1), True, False)
# Calculate the three-candle rolling Low
data['three_candle_Low'] = data['Low'].rolling(3).min()
# Check if the fourth candle is Lower than the Lowest of the previous 3 candles
data['four_candle_Low'] = np.where( data['Low'] <
    data['three_candle_Low'].shift(1), True, False)
# Flag long positions
data['long_positions'] = np.where(data['ATR_breakout'] & data['four_candle_High'], 1, 0)
# Flag short positions
data['short_positions'] = np.where(data['ATR_breakout'] & data['four_candle_Low'], -1, 0)
# Combine long and short  position flags
data['positions'] = data['long_positions'] + data['short_positions']
#print(data['positions'])

current_position = 0
stop_loss = ''
take_profit = ''
entry_time = np.nan
entry_price = np.nan
take_profit_threshold = 0.07
stop_loss_threshold = 0.99

trades = pd.DataFrame()

# Calculate the PnL for exit of a long position
def long_exit(data, time, entry_time, entry_price):
    global trades
    # Rest of the 
    pnl = round(data.loc[time, 'Close'] - entry_price, 2)
    exit_time = str(pd.to_timedelta(data.loc[time, 'ATR'], unit='h'))
    
    # Create a DataFrame object with the trade details
    trade_details = pd.DataFrame([('Long',entry_time,entry_price,exit_time,data.loc[time, 'Close'],pnl)])
    # Add the trade details to the 'trades' DataFrame
    trades = trades.append(trade_details,ignore_index=True)
    return trade_details
   
# Calculate the PnL for exit of a short position

def short_exit(data, time, entry_time, entry_price):
    global trades
    # Rest of the 
    exit_time = str(pd.to_timedelta(data.loc[time, 'ATR'], unit='h'))
    pnl = round(entry_price - data.loc[time, 'Close'], 2)
    # Create a DataFrame object with the trade details
    trade_details = pd.DataFrame([('Short',entry_time,entry_price,exit_time,data.loc[time, 'Close'],pnl)])
    # Add the trade details to the 'trades' DataFrame
    trades = trades.append(trade_details,ignore_index=True)
    return trade_details


for time in tqdm(data.index):
    
    # ---------------------------------------------------------------------------------
    # Long Position
    if (current_position == 0) and (data.loc[time, 'positions'] == 1):
        current_position = 1
        entry_time = data.loc[time, 'Date']
        entry_price = data.loc[time, 'Close']
        stop_loss = entry_price * (1-stop_loss_threshold)
        take_profit = entry_price * (1+take_profit_threshold)
    # ---------------------------------------------------------------------------------
    # Long Exit
    elif (current_position == 1):
        # Check for sl and tp
        if data.loc[time, 'Close'] < stop_loss or data.loc[time, 'Close'] > take_profit:
            trade_details = long_exit(data, time, data.loc[time, 'Date'], entry_price)
            trades = trades.append(trade_details,ignore_index=True)
            current_position = 0
    # ---------------------------------------------------------------------------------
    # Short Position
    if (current_position == 0) and (data.loc[time, 'positions'] == -1):
        current_position = -1
        entry_price = data.loc[time, 'Close']
        stop_loss = entry_price * (1+stop_loss_threshold)
        take_profit = entry_price * (1-take_profit_threshold)
    # ---------------------------------------------------------------------------------
    # Short Exit
    elif (current_position == -1):
        # Check for sl and tp
        if data.loc[time, 'Close'] > stop_loss or data.loc[time, 'Close'] < take_profit:
            trade_details = short_exit(data, time, data.loc[time, 'Date'], entry_price)
            trades = trades.append(trade_details,ignore_index=True)
            current_position = 0
            
trades.to_csv('trades.csv')
# Dataframe showing the details of the each trade in the dataset. 
trades.columns=['Position','Entry Time','Entry Price','Exit Time','Exit Price','PnL']

analytics = pd.DataFrame(index=['ATR + Candle Breakout'])
# Number of long trades
analytics['num_of_long'] = len(trades.loc[trades.Position=='Long'])
# Number of short trades
analytics['num_of_short'] = len(trades.loc[trades.Position=='Short'])
# Total number of trades
analytics['total_trades'] = analytics.num_of_long + analytics.num_of_short
# Profitable trades
analytics['winners'] = len(trades.loc[trades.PnL>0])
# Loss-making trades
analytics['losers'] = len(trades.loc[trades.PnL<=0])
# Win percentage
analytics['win_percentage'] = 100*analytics.winners/analytics.total_trades
# Loss percentage
analytics['loss_percentage'] = 100*analytics.losers/analytics.total_trades
# Per trade profit/loss of winning trades
analytics['per_trade_PnL_winners'] = trades.loc[trades.PnL>0].PnL.mean()
# Per trade profit/loss of losing trades
analytics['per_trade_PnL_losers'] = trades.loc[trades.PnL<=0].PnL.mean()

print(analytics.T)

data['global_PNL'] = np.where(data['positions'] == 1,
data['Close'] - entry_price,
entry_price - data['Close'])

#Sum up all the PnLs
global_PNL = data['global_PNL'].sum()

#Print the global PnL
print("PnL: ", global_PNL)
print(trades)



