import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BollingerBandStrategy:
    def __init__(self, length=20, mult=2.0):
        self.length = length
        self.mult = mult

    def evaluate(self, candle, context):
        # Assuming 'context' is a DataFrame of past candles
        #context = context.append(candle, ignore_index=True)
        new_context = context.copy().append(candle, ignore_index=True)

        if len(new_context) < self.length:
             return None, None, None  # Return None for new_position, upper_band, and lower_band

        close_prices = context['close'][-self.length:]
        basis = np.mean(close_prices)
        dev = self.mult * np.std(close_prices)
        upper = basis + dev
        lower = basis - dev
        current_price = candle['close']

        if current_price > upper:
            return 'short', upper, lower
        elif current_price < lower:
            return 'long', upper, lower
        else:
            return None, upper, lower  # Return None for new_position, but still provide upper and lower bands

class Backtester:
    def __init__(self, timeframe, backtest_days, asset_name, investment_amount, loss_percentage,gain_percentage ):
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
        upper_band = 0.0  # Added upper_band
        lower_band = 0.0  # Added lower_band
        capital = self.investment_amount

        for candle in self.data:
            # Get new position and band values from strategy
            new_position, upper_band, lower_band = strategy.evaluate(candle, context)  

            if position == 'long':
                if candle['close'] < stop_loss:
                    capital -= self.loss_percentage / 100 * capital
                    position = None
                    entry_price = 0.0  # Reset
                    stop_loss = 0.0  # Reset
                elif (candle['close'] - entry_price) / entry_price >= self.gain_percentage / 100:  # Close long position when gain is achieved
                    capital = capital + (candle['close'] - entry_price) / entry_price * capital
                    position = None
                    entry_price = 0.0  # Reset
                    stop_loss = 0.0  # Reset

            elif position == 'short':
                if candle['close'] > stop_loss:
                    capital -= self.loss_percentage / 100 * capital
                    position = None
                    entry_price = 0.0  # Reset
                    stop_loss = 0.0  # Reset
                elif (entry_price - candle['close']) / entry_price >= self.gain_percentage / 100:  # Close short position when gain is achieved
                    capital = capital + (entry_price - candle['close']) / entry_price * capital
                    position = None
                    entry_price = 0.0  # Reset
                    stop_loss = 0.0  # Reset



            if new_position != position and new_position is not None:
                position = new_position
                entry_price = candle['close']
                print(f"Opening {position} position at {entry_price}")

                if position == 'long':
                    stop_loss = entry_price * (1 - self.loss_percentage / 100)
                    upper_band = upper_band  # Memorize the upper band at the moment of buying

                elif position == 'short':
                    stop_loss = entry_price * (1 + self.loss_percentage / 100)
                    lower_band = lower_band  # Memorize the lower band at the moment of selling

            
            #context = context.append(candle, ignore_index=True)
            #context = context.copy().append(candle, ignore_index=True)
            new_context = context.copy().append(candle, ignore_index=True)

            context = new_context.tail(strategy.length)

        print(f"Final capital after backtesting: {capital:.2f}")
        self.context = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.position = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        return capital


    def plot_data(self, strategy):
        context = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'upper', 'lower'])
        position = None
        long_positions = []
        short_positions = []
        long_closes = []
        short_closes = []
        
        for i, candle in enumerate(self.data):
            #new_position = strategy.evaluate(candle, context)
            new_position, upper_band, lower_band = strategy.evaluate(candle, context)  # Unpack the tuple here
            if new_position == 'long':
                long_positions.append(i)
            elif new_position == 'short':
                short_positions.append(i)
            
            # Your strategy logic goes here, similar to the backtest method.
            # Make sure you update `context` DataFrame
                    # Make sure you update `context` DataFrame to have the upper and lower Bollinger Bands
            #context = context.append(candle, ignore_index=True)
            new_context = context.copy().append(candle, ignore_index=True)

            new_context.at[i, 'upper'] = upper_band  # Modified this line
            new_context.at[i, 'lower'] = lower_band  # Modified this line
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

        print("NaN values in 'upper': ", context['upper'].isna().sum())
        print("NaN values in 'lower': ", context['lower'].isna().sum())

        # Debugging: Checking data types
        print("Data type of 'upper': ", context['upper'].dtype)
        print("Data type of 'lower': ", context['lower'].dtype)

        # Potential fix for NaN or None values
        context['upper'].fillna(0, inplace=True)
        context['lower'].fillna(0, inplace=True)

        # Potential fix for type conversion
        context['upper'] = context['upper'].astype('float64')
        context['lower'] = context['lower'].astype('float64')

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
        self.context = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'upper', 'lower'])
        self.position = None
        self.long_positions = []
        self.short_positions = []
        self.long_closes = []
        self.short_closes = []




    
if __name__ == "__main__":
    #timeframe = input("Enter the timeframe (3d,1d, 4h, 1h, 15m, or 1m): ")
    #backtest_days = int(input("Enter the duration for backtesting in days: "))
    # asset_name = input("Enter the asset name (e.g., BTCUSDT): ")
    # investment_amount = float(input("Enter the amount to invest: "))
    #backtest_days = 7
    asset_name = "BTCUSDT"
    investment_amount = 10000
    loss_percentage = float(input("Enter the percentage of loss for stop-loss: "))
    gain_percentage = float(input("Enter the percentage of gain for taking profit: "))
    backtester = Backtester('1d', 200, asset_name, investment_amount, loss_percentage, gain_percentage)
    backtester.fetch_data()
    bollinger_strategy = BollingerBandStrategy()
    backtester.backtest(bollinger_strategy)
    backtester.plot_data(bollinger_strategy)