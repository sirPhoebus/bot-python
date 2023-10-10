import requests
import numpy as np
import pandas as pd

class BollingerBandStrategy:
    def __init__(self, length=20, mult=2.0):
        self.length = length
        self.mult = mult

    def evaluate(self, candle, context):
        # Assuming 'context' is a DataFrame of past candles
        context = context.append(candle, ignore_index=True)
        if len(context) < self.length:
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
    def __init__(self, timeframe, backtest_days, asset_name, investment_amount, loss_percentage):
        self.timeframe = timeframe
        self.backtest_days = backtest_days
        self.asset_name = asset_name
        self.investment_amount = investment_amount
        self.loss_percentage = loss_percentage
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
            new_position, upper_band, lower_band = strategy.evaluate(candle, context)  # Modified this line

            if position == 'long':
                if candle['close'] < stop_loss:
                    capital -= self.loss_percentage / 100 * capital
                    position = None
                elif candle['close'] > upper_band:  # Close long position when price crosses upper band
                    capital = capital + (candle['close'] - entry_price) / entry_price * capital
                    position = None  
                    
            elif position == 'short':
                if candle['close'] > stop_loss:
                    capital -= self.loss_percentage / 100 * capital
                    position = None
                elif candle['close'] < lower_band:  # Close short position when price crosses lower band
                    capital = capital + (entry_price - candle['close']) / entry_price * capital
                    position = None  

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

            # Update context and truncate to keep only the most recent candles.
            context = context.append(candle, ignore_index=True)
            context = context.tail(strategy.length)

        print(f"Final capital after backtesting: {capital}")
        return capital

if __name__ == "__main__":
    # timeframe = input("Enter the timeframe (1d, 4h, 1h, 15m, or 1m): ")
    # backtest_days = int(input("Enter the duration for backtesting in days: "))
    # asset_name = input("Enter the asset name (e.g., BTCUSDT): ")
    # investment_amount = float(input("Enter the amount to invest: "))
    
    backtest_days = 30
    asset_name = "BTCUSDT"
    investment_amount = 10000
    loss_percentage = float(input("Enter the percentage of loss for stop-loss: "))

    backtester = Backtester('15m', backtest_days, asset_name, investment_amount, loss_percentage)
    backtester.fetch_data()

    bollinger_strategy = BollingerBandStrategy()
    backtester.backtest(bollinger_strategy)