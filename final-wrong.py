import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BollingerBandStrategy:
    def __init__(self, length=20, mult=2.0):
        self.counter = 0
        self.last_closed_position = None
        self.length = 20
        self.mult = 2.0

    def calculate_atr(self, context):
        high_low = context['high'] - context['low']
        high_close = np.abs(context['high'] - context['close'].shift())
        low_close = np.abs(context['low'] - context['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.length).mean()
        
        return atr.iloc[-1]
    
    def evaluate(self, candle, context):
        # Assuming 'context' is a DataFrame of past candles
        context = context.append(candle, ignore_index=True)
        if len(context) < self.length:
            return None
        atr = self.calculate_atr(context)
        #print(f"Indicator ATR is : {atr} .")
        if atr > 1000:  
            self.length = 40
            self.mult = 3.0
        elif atr > 800:
            self.length = 30
            self.mult = 2.5
        else:
            self.length = 20
            self.mult = 2.0
        high_prices = context['high'][-self.length:]
        basis = np.mean(high_prices)
        dev = self.mult * np.std(high_prices)
        upper = basis + dev
        lower = basis - dev
        current_price = candle['close']
        
        condition_for_long = current_price < lower
        condition_for_short = current_price > upper

        if self.last_closed_position == 'long':
            if condition_for_short:  # Replace with your actual condition
                self.counter += 1
                if self.counter >= 2:
                    self.counter = 0
                    self.last_closed_position = None  # Resetting the last closed position
                    return 'short'
            else:
                self.counter = 0  # Reset counter if no opposite candles

        elif self.last_closed_position == 'short':
            if condition_for_long:  # Replace with your actual condition
                self.counter += 1
                if self.counter >= 2:
                    self.counter = 0
                    self.last_closed_position = None  # Resetting the last closed position
                    return 'long'
            else:
                self.counter = 0  # Reset counter if no opposite candles
        
        # If last_closed_position is None, your original strategy logic applies
        if condition_for_long:  # Replace with your actual conditions
            return 'long'
        elif condition_for_short:  # Replace with your actual conditions
            return 'short'

        return None  # No position to be opened

class Backtester:
    def __init__(self, timeframe, backtest_days, asset_name, investment_amount, loss_percentage, gain_percentage ):
        self.timeframe = timeframe
        self.backtest_days = backtest_days
        self.asset_name = asset_name
        self.investment_amount = investment_amount
        self.loss_percentage = loss_percentage
        self.gain_percentage = gain_percentage 
        self.data = []

    def fetch_data(self):
        try:
            interval_mapping = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '1d': 1440}
            minutes = self.backtest_days * 24 * 60
            limit = min(int(minutes / interval_mapping[self.timeframe]), 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={self.asset_name}&interval={self.timeframe}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses
            raw_data = response.json()
            for entry in raw_data:
                self.data.append({
                    'timestamp': entry[0],
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': float(entry[5]),
                    # TODO: Calculate lower and upper Bollinger bands here
                })
            print(f"Fetched {len(self.data)} data points.")
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def backtest(self, strategy):
        context = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        position = None
        entry_price = 0.0
        stop_loss = 0.0
        capital = self.investment_amount

        for candle in self.data:
            if position == 'long':
                if candle['close'] < stop_loss:
                    loss_amount = self.loss_percentage / 100 * capital
                    capital -= loss_amount
                    print(f"Closing long position at {candle['close']} due to stop loss. Loss amount: {loss_amount:.2f}")  
                    strategy.last_closed_position = 'long'  
                    position = None
                else:
                    #gain_amount = (candle['close'] - entry_price) / entry_price * capital
                    gain_amount = (candle['close'] - entry_price) * (capital / entry_price)

                    if gain_amount >= self.gain_percentage:
                        print(f"Closing long position at {candle['close']} due to gain. Gain amount: {gain_amount:.2f}%. New capital: {capital:.2f}")  
                        position = None  
                        strategy.last_closed_position = 'long'

            elif position == 'short':
                if candle['close'] > stop_loss:
                    loss_amount = self.loss_percentage / 100 * capital
                    capital -= loss_amount
                    print(f"Closing short position at {candle['close']} due to stop loss. Loss amount: {loss_amount:.2f}")  
                    strategy.last_closed_position = 'short'  
                    position = None
                else:
                    #gain_amount = (entry_price - candle['close']) / entry_price * capital
                    gain_amount = (candle['close'] - entry_price) * (capital / entry_price)

                    if gain_amount >= self.gain_percentage:
                        print(f"Closing short position at {candle['close']} due to gain. Gain amount: {gain_amount:.2f}%. New capital: {capital:.2f}")  
                        position = None  
                        strategy.last_closed_position = 'short'

            new_position = strategy.evaluate(candle, context)
            
            if new_position != position and new_position is not None:
                position = new_position
                entry_price = candle['close']
                print(f"Opening {position} position at {entry_price}")

                if position == 'long':
                    stop_loss = entry_price * (1 - self.loss_percentage / 100)
                elif position == 'short':
                    stop_loss = entry_price * (1 + self.loss_percentage / 100)

            # Update context and truncate to keep only the most recent candles.
            context = context.append(candle, ignore_index=True)
            context = context.tail(strategy.length)

        print(f"Final capital after backtesting: {capital:.2f}")

        return capital

    def plot_data(self, strategy):
        context = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'upper', 'lower'])
        position = None
        long_positions = []
        short_positions = []
        long_closes = []
        short_closes = []
        
        for i, candle in enumerate(self.data):
            new_position = strategy.evaluate(candle, context)
            
            if new_position == 'long':
                long_positions.append(i)
            elif new_position == 'short':
                short_positions.append(i)
            
            # Your strategy logic goes here, similar to the backtest method.
            # Make sure you update `context` DataFrame
                    # Make sure you update `context` DataFrame to have the upper and lower Bollinger Bands
            context = context.append(candle, ignore_index=True)
            if len(context) >= strategy.length:
                high_prices = context['high'][-strategy.length:]
                basis = np.mean(high_prices)
                dev = strategy.mult * np.std(high_prices)
                upper = basis + dev
                lower = basis - dev
                context.at[i, 'upper'] = upper
                context.at[i, 'lower'] = lower

            # Store indices where you close long and short positions, based on your strategy
            if new_position != position and new_position is None:
                if position == 'long':
                    long_closes.append(i)
                elif position == 'short':
                    short_closes.append(i)
            position = new_position

        # Plotting starts here
        plt.figure(figsize=(16, 8))
        plt.plot(context['close'], label='Price', color='black')
        plt.fill_between(context.index, context['lower'], context['upper'], color='lightblue', label='Bollinger Bands')
        plt.plot(context['upper'], label='Upper Bollinger Band', color='blue')
        plt.plot(context['lower'], label='Lower Bollinger Band', color='blue')
        
        # Marking long and short positions on the plot
        plt.scatter(long_positions, context.loc[long_positions, 'close'], marker='o', color='green', label='Long')
        plt.scatter(short_positions, context.loc[short_positions, 'close'], marker='o', color='red', label='Short')

        # Marking the close positions
        plt.scatter(long_closes, context.loc[long_closes, 'close'], marker='x', color='green', label='Close Long')
        plt.scatter(short_closes, context.loc[short_closes, 'close'], marker='x', color='red', label='Close Short')
        
        plt.title(f"{self.asset_name} Bollinger Band Strategy Backtest")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        # This allows you to zoom in on the graph
        plt.grid(True)
        plt.show()



    
if __name__ == "__main__":
    timeframe = input("Enter the timeframe (3d,1d, 4h, 1h, 15m, or 1m): ")
    backtest_days = int(input("Enter the duration for backtesting in days: "))
    # asset_name = input("Enter the asset name (e.g., BTCUSDT): ")
    # investment_amount = float(input("Enter the amount to invest: "))
    gain_percentage = float(input("Enter the percentage of gain to close the position: "))
    #backtest_days = 7
    asset_name = "BTCUSDT"
    investment_amount = 10000
    loss_percentage = float(input("Enter the percentage of loss for stop-loss: "))

    backtester = Backtester('1d', backtest_days, asset_name, investment_amount, loss_percentage, gain_percentage)
    backtester.fetch_data()

    bollinger_strategy = BollingerBandStrategy()
    backtester.backtest(bollinger_strategy)
    backtester.plot_data(bollinger_strategy)