def scalping_strategy(data, stop_loss_threshold, take_profit_threshold):
    # Calculate the Average True Range(ATR)
    data['ATR'] = talib.ATR(data['High'], data['Low'],
                            data['Close'], timeperiod=30)
    # Calculate the rolling mean of ATR
    data['ATR_MA_5'] = data['ATR'].rolling(5).mean()
    
    # Calculate the first minute where ATR breaks out its rolling mean
    data['ATR_breakout'] = np.where((data['ATR'] > data['ATR_MA_5']), True, False)    
    
    # Calculate the three-candle rolling High
    data['three_candle_High'] = data['High'].rolling(3).max()
    # Check if the fourth candle is Higher than the Highest of the previous 3 candles
    data['four_candle_High'] = np.where( data['High'] >
        data['three_candle_High'].shift(1), True, False)
    # Flag the long position signal
    data['long_positions'] = np.where(data['ATR_breakout'] & data['four_candle_High'], 1, 0)
    
    # Calculate the three-candle rolling Low
    data['three_candle_Low'] = data['Low'].rolling(3).min()
    # Check if the fourth candle is Lower than the Lowest of the previous 3 candles
    data['four_candle_Low'] = np.where( data['Low'] <
        data['three_candle_Low'].shift(1), True, False) 
    # Flag the short position signal    
    data['short_positions'] = np.where(data['ATR_breakout'] & data['four_candle_Low'], -1, 0)
    # Combine the long and short flags
    data['positions'] = data['long_positions'] + data['short_positions']
    
    
    current_position = 0
    stop_loss = ''
    take_profit = ''
    entry_price = np.nan
    data['pnl'] = np.nan

    # Calculate the PnL for exit of a long position
    def long_exit(data, time, entry_price):
        pnl = round(data.loc[time, 'Close'] - entry_price, 2)
        data.loc[time, 'pnl'] = pnl
        
    # Calculate the PnL for exit of a short position
    def short_exit(data, time, entry_price):
        pnl = round(entry_price - data.loc[time, 'Close'], 2)
        data.loc[time, 'pnl'] = pnl

    for time in data.index:
        # ---------------------------------------------------------------------------------
        # Long Position
        if (current_position == 0) and (data.loc[time, 'positions'] == 1):
            current_position = 1
            entry_price = data.loc[time, 'Close']
            stop_loss = data.loc[time, 'Close'] * (1-stop_loss_threshold)
            take_profit = data.loc[time, 'Close'] * (1+take_profit_threshold)

        # ---------------------------------------------------------------------------------
        # Long Exit
        elif (current_position == 1):
            # Check for sl and tp
            if data.loc[time, 'Close'] < stop_loss or data.loc[time, 'Close'] > take_profit:
                long_exit(data, time, entry_price)
                current_position = 0

        # ---------------------------------------------------------------------------------
        # Short Position
        if (current_position == 0) and (data.loc[time, 'positions'] == -1):
            current_position = data.loc[time, 'positions']
            entry_price = data.loc[time, 'Close']
            stop_loss = data.loc[time, 'Close'] * (1+stop_loss_threshold)
            take_profit = data.loc[time, 'Close'] * (1-take_profit_threshold)

        # ---------------------------------------------------------------------------------
        # Short Exit
        elif (current_position == -1):
            # Check for sl and tp
            if data.loc[time, 'Close'] > stop_loss or data.loc[time, 'Close'] < take_profit:
                short_exit(data, time, entry_price)
                current_position = 0

        # ---------------------------------------------------------------------------------

                
    return data.pnl.sum()
  

train_test_split = int(data.shape[0]*2/3)
# Get the train data using the split index 
training_yesbank = data.iloc[0:train_test_split].copy()
# Get the test data using the split index 
test_yesbank = data.iloc[train_test_split:].copy()


# Set the range of stop-loss
stop_loss_range = np.arange(0.01,0.08,0.01)
# Set the range of take-profit
take_profit_range = np.arange(0.01,0.08,0.01)

# Empty numpy matrix to PnL for the training data for each combination of stop-loss and take-profit
PnL_grid = np.zeros((len(stop_loss_range),len(take_profit_range)))

max_ = -np.inf
best_params = None

# Iterating over stop-loss and take-profit values
for i,tp in enumerate(take_profit_range):
    for j,sl in enumerate(stop_loss_range):
        PnL_grid[i][j] = scalping_strategy(training_yesbank,sl,tp)
        # If for the current combination PnL is greater than current highest
        # We save the combination and the new high value 
        if PnL_grid[i][j] > max_:
            max_ = PnL_grid[i][j]
            best_params = (sl,tp)

print('The highest PnL: '+str(max_))
print('Optimal stop-loss and profit-taking values: '+str(best_params))