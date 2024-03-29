// *************************************************************************************************************************************************************************************************************************************************************************
// Modified v2.2 Acrypto - Weighted Strategy
// Last update: 04/22/2022
// *************************************************************************************************************************************************************************************************************************************************************************
// *************************************************************************************************************************************************************************************************************************************************************************
// Changes since last update: Bug fix - modified placeholders to specify long or short
// *************************************************************************************************************************************************************************************************************************************************************************

//@version=5
strategy(title='Modified v2.3 Acrypto Weighted Strategy', shorttitle='Acrypto Mod 2.3', overlay=true, precision=5, commission_value=0.06, commission_type=strategy.commission.percent, initial_capital=1000, max_labels_count=500)

// *************************************************************************************************************************************************************************************************************************************************************************
// INPUTS
// *************************************************************************************************************************************************************************************************************************************************************************
// * Type trading
allow_longs = input.bool(true, 'Allow Longs', group='Trading type')
allow_shorts = input.bool(true, 'Allow Shorts', group='Trading type')
// * Datastamp
from_day = input.int(1, 'From Day', minval=1, maxval=31, group='DataStamp')
from_month = input.int(1, 'From Month', minval=1, maxval=12, group='DataStamp')
from_year = input.int(2022, 'From Year', minval=1980, maxval=9999, group='DataStamp')
to_day = input.int(1, 'To Day', minval=1, maxval=31, group='DataStamp')
to_month = input.int(1, 'To Month', minval=1, maxval=12, group='DataStamp')
to_year = input.int(9999, 'To Year', minval=2017, maxval=9999, group='DataStamp')
// leverage
leverage = input.float(1.0, 'Leverage', minval=1, maxval=100, step=0.01, group='Risk Management', inline='line1')
risk = input.float(98.0, 'Wallet Risk %', minval=1, maxval=100, step=1, group='Risk Management', inline='line1')/100
inverse = input.bool(false, 'Inverse Perp?', group= 'Risk Management', inline='line1')
// * Stop loss
stoploss = input.bool(true, 'Stop loss', group='Stop loss')
movestoploss = input.string('Percentage', 'Move stop loss', options=['None', 'Percentage', 'TP-1', 'TP-2', 'TP-3'], group='Stop loss')
movestoploss_entry = input.bool(true, 'Move stop loss to entry', group='Stop loss')
stoploss_perc = input.float(5.6, 'Stop Loss %', minval=0, maxval=100, group='Stop loss') * 0.01
move_stoploss_factor = input.float(20.0, 'Move stop loss factor %', group='Stop loss') * 0.01 + 1
stop_source = input.source(close, 'Stop Source', group='Stop loss')
// * Take profits
take_profits = input.bool(true, 'Take profits', group='Take Profits')
// retrade= input.bool(false, 'Retrade', group='Take Profits')
MAX_TP = input.int(3, 'Max number of TP', minval=1, maxval=6, group='Take Profits')
long_profit_perc = input.float(6.9, 'Long - take profits each x (%)', minval=0.0, maxval=999, step=1, group='Take Profits') * 0.01
long_profit_qty = input.float(49, 'Long - take x (%) from open position', minval=0.0, maxval=100, step=1, group='Take Profits')
short_profit_perc = input.float(15, 'Short - take profits each x (%)', minval=0.0, maxval=999, step=1, group='Take Profits') * 0.01
short_profit_qty = input.float(17, 'Short - take x (%) from open position', minval=0.0, maxval=100, step=1, group='Take Profits')
// * Delays
delay_macd = input.int(1, 'Candles delay MACD', minval=1, group='Delays')
delay_srsi = input.int(2, 'Candles delay Stoch RSI', minval=1, group='Delays')
delay_rsi = input.int(2, 'Candles delay RSI', minval=1, group='Delays')
delay_super = input.int(1, 'Candles delay Supertrend', minval=1, group='Delays')
delay_cross = input.int(1, 'Candles delay MA cross', minval=1, group='Delays')
delay_exit = input.int(7, 'Candles delay exit', minval=1, group='Delays')
// * Inputs Weigthed strategies
str_0 = input.bool(true, 'Strategy 0: Weighted Strategy', group='Weights')
weight_trigger = input.int(2, 'Weight Signal entry [0, 5]', minval=0, maxval=5, step=1, group='Weights')
weight_str1 = input.int(1, 'Weight Strategy 1 [0, 5]', minval=0, maxval=5, step=1, group='Weights')
weight_str2 = input.int(1, 'Weight Strategy 2 [0, 5]', minval=0, maxval=5, step=1, group='Weights')
weight_str3 = input.int(1, 'Weight Strategy 3 [0, 5]', minval=0, maxval=5, step=1, group='Weights')
weight_str4 = input.int(1, 'Weight Strategy 4 [0, 5]', minval=0, maxval=5, step=1, group='Weights')
weight_str5 = input.int(1, 'Weight Strategy 5 [0, 5]', minval=0, maxval=5, step=1, group='Weights')
// * Inputs strategy 1: MACD 
str_1 = input.bool(true, 'Strategy 1: MACD', group='Strategy 1: MACD')
MA1_period_1 = input.int(12, 'MA 1', minval=1, maxval=9999, step=1, group='Strategy 1: MACD')
MA1_type_1 = input.string('SMA', 'MA1 Type', options=['RMA', 'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'VWMA'], group='Strategy 1: MACD')
MA1_source_1 = input.source(high, 'MA1 Source', group='Strategy 1: MACD')
MA2_period_1 = input.int(24, 'MA 2', minval=1, maxval=9999, step=1, group='Strategy 1: MACD')
MA2_type_1 = input.string('WMA', 'MA2 Type', options=['RMA', 'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'VWMA'], group='Strategy 1: MACD')
MA2_source_1 = input.source(high, 'MA2 Source', group='Strategy 1: MACD')
// * Inputs strategy 2: Stoch RSI oversold/overbought
str_2 = input.bool(true, 'Strategy 2: Stoch RSI', group='Strategy 2: Stoch RSI')
long_RSI = input.float(70, 'Exit SRSI Long (%)', minval=0.0, step=1, group='Strategy 2: Stoch RSI')
short_RSI = input.float(27, 'Exit SRSI Short (%)', minval=0.0, step=1, group='Strategy 2: Stoch RSI')
length_RSI = input.int(14, 'RSI Length', group='Strategy 2: Stoch RSI')
length_stoch = input.int(14, 'RSI Stochastic', group='Strategy 2: Stoch RSI')
smoothK = input.int(3, 'Smooth k', group='Strategy 2: Stoch RSI')
// * Inputs strategy 3: RSI oversold/overbought
str_3 = input.bool(true, 'Strategy 3: RSI', group='Strategy 3: RSI')
long_RSI2 = input.float(77, 'Exit RSI Long (%)', minval=0.0, step=1, group='Strategy 3: RSI')
short_RSI2 = input.float(30, 'Exit RSI Short (%)', minval=0.0, step=1, group='Strategy 3: RSI')
// * Inputs strategy 4: Supertrend
str_4 = input.bool(true, 'Strategy 4: Supertrend', group='Strategy 4: Supertrend')
periods_4 = input.int(2, 'ATR Period', group='Strategy 4: Supertrend')
source_4 = input.source(hl2, 'Source', group='Strategy 4: Supertrend')
multiplier = input.float(2.4, 'ATR Multiplier', step=0.1, group='Strategy 4: Supertrend')
change_ATR = input.bool(true, 'Change ATR Calculation Method ?', group='Strategy 4: Supertrend')
// * Inputs strategy 5: MA CROSS
str_5 = input.bool(true, 'Strategy 5: MA CROSS', group='Strategy 5: MA CROSS')
MA1_period_5 = input.int(12, 'MA_1', minval=1, maxval=9999, step=1, group='Strategy 5: MA CROSS')
MA1_type_5 = input.string('SMA', 'MA_1 Type', options=['RMA', 'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'VWMA'], group='Strategy 5: MA CROSS')
MA1_source_5 = input.source(close, 'MA_1 Source', group='Strategy 5: MA CROSS')
MA2_period_5 = input.int(24, 'MA_2', minval=1, maxval=9999, step=1, group='Strategy 5: MA CROSS')
MA2_type_5 = input.string('EMA', 'MA_2 Type', options=['RMA', 'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'VWMA'], group='Strategy 5: MA CROSS')
MA2_source_5 = input.source(close, 'MA_2 Source', group='Strategy 5: MA CROSS')
// * Inputs Potential TOP/BOTTOM
str_6 = input.bool(true, 'Close order at potential Top/Bottom', group='Potential TOP/BOTTOM')
top_qty = input.float(30, 'Top - take x (%) from the remaining position', minval=0.0, maxval=100, step=1, group='Potential TOP/BOTTOM')
bottom_qty = input.float(30, 'Bottom - take x (%) from the remaining position', minval=0.0, maxval=100, step=1, group='Potential TOP/BOTTOM') 
source_6_top = input.source(close, 'TP-TOP at previous?', group='Potential TOP/BOTTOM')
source_6_bottom = input.source(close, 'TP-BOTTOM at previous?', group='Potential TOP/BOTTOM')
long_trail_perc = input.float(150, 'Trail volume Long (%)', minval=0.0, step=1, group='Potential TOP/BOTTOM') * 0.01
short_trail_perc = input.float(150, 'Trail volume Short (%)', minval=0.0, step=1, group='Potential TOP/BOTTOM') * 0.01
// * Miscellaneous
show_signals = input.bool(true, 'Show Buy/Sell Signals ?', group='Miscellaneous')
// * Alarms
alarm_label_long = input.text_area('open long message', 'Alert message - open long', group='Basic alert system')
alarm_label_short = input.text_area('open short message', 'Alert message - open short', group='Basic alert system')
alarm_label_close_long = input.text_area('close long message', 'Alert message - close long', group='Basic alert system')
alarm_label_close_short = input.text_area('close short message', 'Alert message - close short', group='Basic alert system')
alarm_label_TP_long = input.text_area('Long TP 1-6 message', 'Alert message - Long Take Profit', group='Basic alert system')
alarm_label_TP_short = input.text_area('Short TP 1-6 message', 'Alert message - Short Take Profit', group='Basic alert system')
alarm_label_TP_T = input.text_area('TP-T message', 'Alert message - Take Profit - Top', group='Basic alert system')
alarm_label_TP_B = input.text_area('TP-B message', 'Alert message - Take Profit - Bottom', group='Basic alert system')

// *************************************************************************************************************************************************************************************************************************************************************************
// GLOBAL VARIABLES
// *************************************************************************************************************************************************************************************************************************************************************************
start = timestamp(from_year, from_month, from_day, 00, 00) // backtest start window
end = timestamp(to_year, to_month, to_day, 23, 59)// backtest finish window
var FLAG_FIRST = false
var price_stop_long = 0.
var price_stop_short = 0.
var profit_qty = 0. // Quantity to close per TP from open position
var profit_perc = 0. // Percentage to take profits since open position or last TP
var nextTP = 0. // Next target to take profits
var num_tp_left = 0 // number of take profits that havent hit yet
var since_entry = 0 // Number of bars since open last postion
var since_close = 0 // Number of bars since close or TP/STOP last position
var float risk_pct = na // The % of balance to use for market order when opening new position (calculated as leverage * risk)
var float lev_risk_pct = na
var string lbl = na

var tp1_pct_long = long_profit_perc * 100
var tp2_pct_long = long_profit_perc * 200
var tp3_pct_long = long_profit_perc * 300
var tp4_pct_long = long_profit_perc * 400
var tp5_pct_long = long_profit_perc * 500
var tp6_pct_long = long_profit_perc * 600

var tp1_pct_short = short_profit_perc * 100
var tp2_pct_short = short_profit_perc * 200
var tp3_pct_short = short_profit_perc * 300
var tp4_pct_short = short_profit_perc * 400
var tp5_pct_short = short_profit_perc * 500
var tp6_pct_short = short_profit_perc * 600

var limit_pos_pct_long = long_profit_qty
var limit_pos_pct_short = short_profit_qty

var ext_signal = 0



// *************************************************************************************************************************************************************************************************************************************************************************
// FUNCTIONS
// *************************************************************************************************************************************************************************************************************************************************************************

// * Function to replace placeholders with variable values
replace_str(the_str) =>
    string str_formatted = the_str
    float pct_available_long = 100 - (num_tp_left * limit_pos_pct_long) // The % of position that is not in limit orders for tp 1-6
    float pct_available_short = 100 - (num_tp_left * limit_pos_pct_short) // The % of position that is not in limit orders for tp 1-6
    float market_pos_pct_long = top_qty >= pct_available_long ? pct_available_long : top_qty // The % of position to use in market orders for tp-t/tp-b
    float market_pos_pct_short = bottom_qty >= pct_available_short ? pct_available_short : bottom_qty // The % of position to use in market orders for tp-t/tp-b
    
    str_formatted := str.replace_all(str_formatted, '#label#', str.tostring(lbl))
    str_formatted := str.replace_all(str_formatted, '#symbol#', str.tostring(syminfo.ticker))
    str_formatted := str.replace_all(str_formatted, '#leverage#', str.tostring(leverage))
    str_formatted := str.replace_all(str_formatted, '#risk_pct#', str.tostring(risk_pct))
    str_formatted := str.replace_all(str_formatted, '#lev_risk_pct#', str.tostring(lev_risk_pct))
    str_formatted := str.replace_all(str_formatted, '#market_pos_pct_long#', str.tostring(market_pos_pct_long))
    str_formatted := str.replace_all(str_formatted, '#market_pos_pct_short#', str.tostring(market_pos_pct_short))
    str_formatted := str.replace_all(str_formatted, '#limit_pos_pct_long#', str.tostring(limit_pos_pct_long))
    str_formatted := str.replace_all(str_formatted, '#limit_pos_pct_short#', str.tostring(limit_pos_pct_short))
    str_formatted := str.replace_all(str_formatted, '#sl_pct#', str.tostring(stoploss_perc*100))
    str_formatted := str.replace_all(str_formatted, '#tp1_pct_long#', str.tostring(tp1_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp2_pct_long#', str.tostring(tp2_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp3_pct_long#', str.tostring(tp3_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp4_pct_long#', str.tostring(tp4_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp5_pct_long#', str.tostring(tp5_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp6_pct_long#', str.tostring(tp6_pct_long))
    str_formatted := str.replace_all(str_formatted, '#tp1_pct_short#', str.tostring(tp1_pct_short)) // The % profit for tp1 short
    str_formatted := str.replace_all(str_formatted, '#tp2_pct_short#', str.tostring(tp2_pct_short))
    str_formatted := str.replace_all(str_formatted, '#tp3_pct_short#', str.tostring(tp3_pct_short))
    str_formatted := str.replace_all(str_formatted, '#tp4_pct_short#', str.tostring(tp4_pct_short))
    str_formatted := str.replace_all(str_formatted, '#tp5_pct_short#', str.tostring(tp5_pct_short))
    str_formatted := str.replace_all(str_formatted, '#tp6_pct_short#', str.tostring(tp6_pct_short))
    // str_formatted := str.replace_all(str_formatted, '#short_sl_pct#', str.tostring(tp6_pct_short))
    // str_formatted := str.replace_all(str_formatted, '#long_sl_price#', str.tostring(tp6_pct_short))
    // str_formatted := str.replace_all(str_formatted, '#short_sl_price#', str.tostring(tp6_pct_short))
    str_formatted

// * MA type
ma(MAType, MASource, MAPeriod) =>
    switch MAType
        'SMA' => ta.sma(MASource, MAPeriod)
        'EMA' => ta.ema(MASource, MAPeriod)
        'WMA' => ta.wma(MASource, MAPeriod)
        'RMA' => ta.rma(MASource, MAPeriod)
        'HMA' => ta.wma(2 * ta.wma(MASource, MAPeriod / 2) - ta.wma(MASource, MAPeriod), math.round(math.sqrt(MAPeriod)))
        'DEMA' =>
            e = ta.ema(MASource, MAPeriod)
            2 * e - ta.ema(e, MAPeriod)
        'TEMA' =>
            e = ta.ema(MASource, MAPeriod)
            3 * (e - ta.ema(e, MAPeriod)) + ta.ema(ta.ema(e, MAPeriod), MAPeriod)
        'VWMA' => ta.vwma(MASource, MAPeriod)

// *************************************************************************************************************************************************************************************************************************************************************************
// * Number strategies
// *************************************************************************************************************************************************************************************************************************************************************************
n_strategies() =>
    var result = 0.
    if str_1
        result := 1.
    if str_2
        result += 1.
    if str_3
        result += 1.
    if str_4
        result += 1.
    if str_5
        result += 1.
// *************************************************************************************************************************************************************************************************************************************************************************
// * Price take profit
// *************************************************************************************************************************************************************************************************************************************************************************
price_takeProfit(percentage, N) =>
    if strategy.position_size > 0
        strategy.position_avg_price * (1 + N * percentage)
    else
        strategy.position_avg_price * (1 - N * percentage)
// *************************************************************************************************************************************************************************************************************************************************************************
// * Weigthed values
// *************************************************************************************************************************************************************************************************************************************************************************
weight_values(signal) =>
    if signal
        weight = 1.0
    else
        weight = 0.
// *************************************************************************************************************************************************************************************************************************************************************************
// * Weigthed total
// *************************************************************************************************************************************************************************************************************************************************************************
weight_total(signal1, signal2, signal3, signal4, signal5) =>
    weight_str1 * weight_values(signal1) + weight_str2 * weight_values(signal2) + weight_str3 * weight_values(signal3) + weight_str4 * weight_values(signal4) + weight_str5 * weight_values(signal5)
// *************************************************************************************************************************************************************************************************************************************************************************
// * Color
// *************************************************************************************************************************************************************************************************************************************************************************
colors(type, value=0) =>
    switch str.lower(type)
        'buy'=> color.new(color.aqua, value)
        'sell' => color.new(color.gray, value)
        'TP' => color.new(color.aqua, value)
        'SL' => color.new(color.gray, value)
        'signal' => color.new(color.orange, value)
        'profit' => color.new(color.teal, value)
        'loss' => color.new(color.red, value)
        'info' => color.new(color.white, value)
        'highlights' => color.new(color.orange, value)
// *************************************************************************************************************************************************************************************************************************************************************************
// * Bar since last entry
// *************************************************************************************************************************************************************************************************************************************************************************
bars_since_entry() =>
    bar_index - strategy.opentrades.entry_bar_index(0)
// *************************************************************************************************************************************************************************************************************************************************************************
// * Bar since close or TP/STOP
// *************************************************************************************************************************************************************************************************************************************************************************
bars_since_close() =>
    ta.barssince(ta.change(strategy.closedtrades))
// *************************************************************************************************************************************************************************************************************************************************************************
// ADDITIONAL GLOBAL VARIABLES
// *************************************************************************************************************************************************************************************************************************************************************************
// * Compute time since last entry and last close/TP position
since_entry := bars_since_entry()
since_close := bars_since_close()
if strategy.opentrades == 0
    since_entry := delay_exit
if strategy.closedtrades == 0
    since_close := delay_exit
    
// * Compute profit quantity and profit percentage
if strategy.position_size > 0 //and since_entry == 0
    profit_qty := long_profit_qty
    profit_perc := long_profit_perc
else if strategy.position_size < 0 //and since_entry == 0
    profit_qty := short_profit_qty
    profit_perc := short_profit_perc
else
    nextTP := 0. // Next Take Profit target (out of market)

// *************************************************************************************************************************************************************************************************************************************************************************
// STRATEGIES
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 1: MACD
// *************************************************************************************************************************************************************************************************************************************************************************
MA1 = ma(MA1_type_1, MA1_source_1, MA1_period_1)
MA2 = ma(MA2_type_1, MA2_source_1, MA2_period_1)

MACD = MA1 - MA2
signal = ma('SMA', MACD, 9)
trend= MACD - signal

long = MACD > signal
short = MACD < signal
proportion = math.abs(MACD / signal)

// * Conditions
long_signal1 = long and long[delay_macd - 1] and not long[delay_macd]
short_signal1 = short and short[delay_macd - 1] and not short[delay_macd]
close_long1 = short and not long[delay_macd]
close_short1 = long and not short[delay_macd]
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 2: STOCH RSI
// *************************************************************************************************************************************************************************************************************************************************************************
rsi = ta.rsi(close, length_RSI)
srsi = ta.stoch(rsi, rsi, rsi, length_stoch)
k = ma('SMA', srsi, smoothK)
isRsiOB = k >= long_RSI
isRsiOS = k <= short_RSI

// * Conditions
long_signal2 = isRsiOS[delay_srsi] and not isRsiOB and since_entry >= delay_exit and since_close >= delay_exit
short_signal2 = isRsiOB[delay_srsi] and not isRsiOS and since_entry >= delay_exit and since_close >= delay_exit
close_long2 = short_signal2
close_short2 = long_signal2
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 3: RSI
// *************************************************************************************************************************************************************************************************************************************************************************
isRsiOB2 = rsi >= long_RSI2
isRsiOS2 = rsi <= short_RSI2

// * Conditions
long_signal3 = isRsiOS2[delay_rsi] and not isRsiOB2 and since_entry >= delay_exit and since_close >= delay_exit
short_signal3 = isRsiOB2[delay_rsi] and not isRsiOS2 and since_entry >= delay_exit and since_close >= delay_exit
close_long3 = short_signal3
close_short3 = long_signal3
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 4: SUPERTREND
// *************************************************************************************************************************************************************************************************************************************************************************
atr2 = ma('SMA', ta.tr, periods_4)
atr = change_ATR ? ta.atr(periods_4) : atr2
up = source_4 - multiplier * atr
up1 = nz(up[1], up)
up := close[1] > up1 ? math.max(up, up1) : up

dn = source_4 + multiplier * atr
dn1 = nz(dn[1], dn)
dn := close[1] < dn1 ? math.min(dn, dn1) : dn

trend := 1
trend := nz(trend[1], trend)
trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend

// * Conditions
long4 = trend == 1
short4 = trend == -1
long_signal4 = trend == 1 and trend[delay_super - 1] == 1 and trend[delay_super] == -1
short_signal4 = trend == -1 and trend[delay_super - 1] == -1 and trend[delay_super] == 1
changeCond = trend != trend[1]
close_long4 = short_signal4
close_short4 = short_signal4
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 5: MA CROSS
// *************************************************************************************************************************************************************************************************************************************************************************
MA12 = ma(MA1_type_5, MA1_source_5, MA1_period_5)
MA22 = ma(MA2_type_5, MA2_source_5, MA2_period_5)

long5 = MA12 > MA22
short5 = MA12 < MA22

// * Conditions
long_signal5 = long5 and long5[delay_cross - 1] and not long5[delay_cross]
short_signal5 = short5 and short5[delay_cross - 1] and not short5[delay_cross]
close_long5 = short5 and not long5[delay_cross]
close_short5 = long5 and not short5[delay_cross]
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 6: POTENTIAL TOP/BOTTOM
// *************************************************************************************************************************************************************************************************************************************************************************
// * Combination RSI, Stoch RSI, MACD, volume, and weighted-strategy to detect potential TOP/BOTTOMS areas
volumeRSI_condition = volume[2] > volume[3] and volume[2] > volume[4] and volume[2] > volume[5]
condition_OB1 = isRsiOB2 and (isRsiOB or volume < ma('SMA', volume, 20) / 2) and volumeRSI_condition
condition_OS1 = isRsiOS2 and (isRsiOS or volume < ma('SMA', volume, 20) / 2) and volumeRSI_condition

condition_OB2 = volume[2] / volume[1] > (1.0 + long_trail_perc) and isRsiOB and volumeRSI_condition
condition_OS2 = volume[2] / volume[1] > (1.0 + short_trail_perc) and isRsiOS and volumeRSI_condition

condition_OB3 = weight_total(MACD < signal, isRsiOB, isRsiOB2, short4, short5) >= weight_trigger
condition_OS3 = weight_total(MACD > signal, isRsiOS, isRsiOS2, long4, long5) >= weight_trigger

condition_OB = (condition_OB1 or condition_OB2)
condition_OS = (condition_OS1 or condition_OS2)
condition_OB_several = condition_OB[1] and condition_OB[2] or condition_OB[1] and condition_OB[3] or condition_OB[1] and condition_OB[4] or condition_OB[1] and condition_OB[5] or condition_OB[1] and condition_OB[6] or condition_OB[1] and condition_OB[7] 
condition_OS_several = condition_OS[1] and condition_OS[2] or condition_OS[1] and condition_OS[3] or condition_OS[1] and condition_OS[4] or condition_OS[1] and condition_OS[5] or condition_OS[1] and condition_OS[6] or condition_OS[1] and condition_OS[7] 
// *************************************************************************************************************************************************************************************************************************************************************************
// STRATEGY ENTRIES AND EXITS
// *************************************************************************************************************************************************************************************************************************************************************************
margin = strategy.equity * risk
risk_pct := risk * 100
lev_risk_pct := leverage * risk * 100
ord_size = inverse ? math.round(leverage * margin,3) : math.round((margin*leverage)/close,3)

if time >= start and time <= end
    // ***************************************************************************************************************************************************************************
    // * Set Entries
    // ***************************************************************************************************************************************************************************
    if str_0
        if not str_1
            weight_str1 := 0
        if not str_2
            weight_str2 := 0
        if not str_3
            weight_str3 := 0
        if not str_4
            weight_str4 := 0
        if not str_5
            weight_str5 := 0
        if allow_shorts == true
            w_total = weight_total(short_signal1, short_signal2, short_signal3, short_signal4, short_signal5)
            if w_total >= weight_trigger and strategy.position_size >= 0
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, alert_message=replace_str(alarm_label_short))
        if allow_longs == true
            w_total = weight_total(long_signal1, long_signal2, long_signal3, long_signal4, long_signal5)
            if w_total >= weight_trigger and strategy.position_size <= 0
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, alert_message=replace_str(alarm_label_long))
    else
        if allow_shorts == true and strategy.position_size >= 0
            if str_1
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, when=short_signal1, alert_message=replace_str(alarm_label_short))
            if str_2
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, when=short_signal2, alert_message=replace_str(alarm_label_short))
            if str_3
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, when=short_signal3, alert_message=replace_str(alarm_label_short))
            if str_4
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, when=short_signal4, alert_message=replace_str(alarm_label_short))
            if str_5
                lbl := 'Short Entry'
                strategy.entry('Short', strategy.short, qty=ord_size, when=short_signal5, alert_message=replace_str(alarm_label_short))
        if allow_longs == true and strategy.position_size <= 0
            if str_1
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, when=long_signal1, alert_message=replace_str(alarm_label_long))
            if str_2
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, when=long_signal2, alert_message=replace_str(alarm_label_long))
            if str_3
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, when=long_signal3, alert_message=replace_str(alarm_label_long))
            if str_4
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, when=long_signal4, alert_message=replace_str(alarm_label_long))
            if str_5
                lbl := 'Long Entry'
                strategy.entry('Long', strategy.long, qty=ord_size, when=long_signal5, alert_message=replace_str(alarm_label_long))
    // ***************************************************************************************************************************************************************************
    // * Set Take Profits
    // ***************************************************************************************************************************************************************************
    if strategy.position_size > 0 and take_profits and since_entry == 0
        for i = 1 to MAX_TP
            id = 'TP ' + str.tostring(i)
            lbl := 'Long ' + id
            strategy.exit(id=id, limit=price_takeProfit(profit_perc, i), qty_percent=profit_qty, comment=id, alert_message=replace_str(alarm_label_TP_long))
    if strategy.position_size < 0 and take_profits and since_entry == 0
        for i = 1 to MAX_TP
            id = 'TP ' + str.tostring(i)
            lbl := 'Short ' + id
            strategy.exit(id=id, limit=price_takeProfit(profit_perc, i), qty_percent=profit_qty, comment=id, alert_message=replace_str(alarm_label_TP_short))
    // ***************************************************************************************************************************************************************************
    // * Set Stop loss
    // ***************************************************************************************************************************************************************************
    if strategy.position_size > 0
        if since_close == 0
            if high > price_takeProfit(profit_perc, 6) and MAX_TP >= 6
                n = 6
                nextTP := na
                num_tp_left := na
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_long := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_long := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_long := price_takeProfit(profit_perc, n-3)
            else if high > price_takeProfit(profit_perc, 5) and MAX_TP >= 5
                n = 5
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_long := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_long := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_long := price_takeProfit(profit_perc, n-3)
            else if high > price_takeProfit(profit_perc, 4) and MAX_TP >= 4
                n = 4
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_long := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_long := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_long := price_takeProfit(profit_perc, n-3)
            else if high > price_takeProfit(profit_perc, 3) and MAX_TP >= 3
                n = 3
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_long := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_long := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
            else if high > price_takeProfit(profit_perc, 2) and MAX_TP >= 2
                n = 2
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_long := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
            else if high > price_takeProfit(profit_perc, 1) and MAX_TP >= 1
                n = 1
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_long := strategy.position_avg_price * (1 + n*profit_perc - stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
                else if movestoploss == 'TP-2' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_long := strategy.position_avg_price
        if since_entry == 0
            n = 0
            nextTP := price_takeProfit(profit_perc, n + 1)
            num_tp_left := MAX_TP - n
            price_stop_long := strategy.position_avg_price * (1 - stoploss_perc) 
    if strategy.position_size < 0
        if since_close == 0
            if low < price_takeProfit(profit_perc, 6) and MAX_TP >= 6
                n = 6
                nextTP := na
                num_tp_left := na
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_short := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_short := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_short := price_takeProfit(profit_perc, n-3)
            else if low < price_takeProfit(profit_perc, 5) and MAX_TP >= 5
                n = 5
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_short := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_short := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_short := price_takeProfit(profit_perc, n-3)
            else if low < price_takeProfit(profit_perc, 4) and MAX_TP >= 4
                n = 4
                nextTP := price_takeProfit(profit_perc, n + 1)
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_short := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_short := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3'
                    price_stop_short := price_takeProfit(profit_perc, n-3)
            else if low < price_takeProfit(profit_perc, 3) and MAX_TP >= 3
                n = 3
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_short := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2'
                    price_stop_short := price_takeProfit(profit_perc, n-2)
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
            else if low < price_takeProfit(profit_perc, 2) and MAX_TP >= 2
                n = 2
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1'
                    price_stop_short := price_takeProfit(profit_perc, n-1)
                else if movestoploss == 'TP-2' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
            else if low < price_takeProfit(profit_perc, 1) and MAX_TP >= 1
                n = 1 
                nextTP := price_takeProfit(profit_perc, n + 1)
                num_tp_left := MAX_TP - n
                if movestoploss == 'Percentage'
                    price_stop_short := strategy.position_avg_price * (1 - n*profit_perc + stoploss_perc * move_stoploss_factor)
                else if movestoploss == 'TP-1' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
                else if movestoploss == 'TP-2' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
                else if movestoploss == 'TP-3' and movestoploss_entry
                    price_stop_short := strategy.position_avg_price
        if since_entry == 0
            n = 0
            nextTP := price_takeProfit(profit_perc, n + 1)
            num_tp_left := MAX_TP - n
            price_stop_short := strategy.position_avg_price * (1 + stoploss_perc)
    // ***************************************************************************************************************************************************************************
    // * Set Exits
    // ***************************************************************************************************************************************************************************
    if allow_longs == true and allow_shorts == false
        if str_0
            w_total = weight_total(short_signal1, short_signal2, short_signal3, short_signal4, short_signal5)
            lbl := 'Close Long'
            strategy.close('Long', when=w_total>=weight_trigger, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
        else
            if str_1
                lbl := 'Close Long'
                close_long1 := close - (close * 1.02)
                strategy.close('Long', when=close_long1, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
            if str_2
                lbl := 'Close Long'
                close_long2 := close - (close * 1.02)
                strategy.close('Long', when=close_long2, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
            if str_3
                lbl := 'Close Long'
                close_long3 := close - (close * 1.02)
                strategy.close('Long', when=close_long3, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
            if str_4
                lbl := 'Close Long'
                close_long4 := close - (close * 1.02)
                strategy.close('Long', when=close_long4, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
            if str_5
                lbl := 'Close Long'
                close_long5 := close - (close * 1.02)
                strategy.close('Long', when=close_long5, qty_percent=100, comment='SHORT', alert_message=replace_str(alarm_label_close_long))
    if allow_longs == false and allow_shorts == true
        if str_0
            w_total = weight_total(long_signal1, long_signal2, long_signal3, long_signal4, long_signal5)
            lbl := 'Close Short'
            strategy.close('Short', when=w_total>=weight_trigger, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
        else
            if str_1
                lbl := 'Close Short'
                close_long1 := close + (close * 1.02)
                strategy.close('Short', when=close_long1, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
            if str_2
                lbl := 'Close Short'
                close_long2 := close + (close * 1.02)
                strategy.close('Short', when=close_long2, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
            if str_3
                lbl := 'Close Short'
                close_long3 := close + (close * 1.02)
                strategy.close('Short', when=close_long3, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
            if str_4
                lbl := 'Close Short'
                close_long4 := close + (close * 1.02)
                strategy.close('Short', when=close_long4, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
            if str_5
                lbl := 'Close Short'
                close_long5 := close + (close * 1.02)
                strategy.close('Short', when=close_long5, qty_percent=100, comment='LONG', alert_message=replace_str(alarm_label_close_short))
    if allow_shorts == true and strategy.position_size < 0 and stoploss and since_entry > 0
        lbl := 'Close Short'
        strategy.close('Short', when=stop_source >= price_stop_short, qty_percent=100, comment='STOP', alert_message=replace_str(alarm_label_close_short))
        if str_6
            if top_qty == 100
                lbl := 'Close Short'
                strategy.close('Short', when=condition_OS_several, qty_percent=bottom_qty, comment='STOP', alert_message=replace_str(alarm_label_close_short))
            else    
                lbl := 'TP-B Short'
                strategy.exit('Short', when=condition_OS_several, limit=source_6_bottom[1], qty_percent=bottom_qty, comment='TP-B', alert_message=replace_str(alarm_label_TP_B))
    if allow_longs == true and strategy.position_size > 0 and stoploss and since_entry > 0
        lbl := 'Close Long'
        strategy.close('Long', when=stop_source <= price_stop_long, qty_percent=100, comment='STOP', alert_message=replace_str(alarm_label_close_long))
        if str_6
            if top_qty == 100
                lbl := 'Close Long'
                strategy.close('Long', when=condition_OB_several, qty_percent=top_qty, comment='STOP', alert_message=replace_str(alarm_label_close_long))
            else
                lbl := 'TP-T Long'
                strategy.exit('Long', when=condition_OB_several, limit=source_6_top[1], qty_percent=top_qty, comment='TP-T', alert_message=replace_str(alarm_label_TP_T))
// *************************************************************************************************************************************************************************************************************************************************************************
// * Data window - debugging
// *************************************************************************************************************************************************************************************************************************************************************************
price_stop = strategy.position_size > 0 ? price_stop_long : price_stop_short

plotchar(volume[2] / volume[1], "Volume 2 / Volume 1", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar(since_entry, "Since entry [bars]", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar(since_close, "Since close/TP [bars]", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar(strategy.position_avg_price, "Average position price", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar(strategy.position_size, "Position size", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar((math.abs(strategy.position_size[1] - strategy.position_size) / math.abs(strategy.position_size[1])) * 100, 'Difference in size', '', location.top, size=size.tiny, color=color.gray)
plotchar(profit_qty, 'profit_qty', '', location.top, size=size.tiny, color=color.gray)
plotchar(num_tp_left, 'num_tp_left', '', location.top, size=size.tiny, color=color.gray)
plotchar(nextTP, "Next TP target", "", location.top, size = size.tiny, color=color.new(color.teal, 0))
plotchar(risk_pct, 'risk_pct', '', location.top, size=size.tiny, color=color.gray)
plotchar(price_stop, "STOP Price", "", location.top, size = size.tiny, color=color.new(color.gray, 0))
plotchar(strategy.opentrades, "Open trades", "", location.top, size = size.tiny, color=strategy.position_size > 0 ? color.blue : color.gray)
plotchar(strategy.netprofit, "Net profit [$]", "", location.top, size = size.tiny, color=strategy.netprofit > 0 ? color.blue : color.gray)
plotchar(strategy.grossprofit, "Gross profit [$]", "", location.top, size = size.tiny, color=color.blue) 
plotchar(strategy.grossloss, "Gross loss [$]", "", location.top, size = size.tiny, color=color.gray) 
plotchar(strategy.openprofit, "Unrealized P&L [$]", "", location.top, size = size.tiny, color=strategy.openprofit > 0 ? color.blue : color.gray)
plotchar(strategy.closedtrades, "Closed trades", "", location.top, size = size.tiny, color=color.orange)
plotchar(strategy.closedtrades.commission(strategy.closedtrades-1), 'Closed trade Commission', '', location.top, size=size.tiny, color=color.fuchsia)
plotchar(strategy.wintrades/strategy.closedtrades*100, "Winrate [%]", "", location.top, size = size.tiny, color=strategy.wintrades/strategy.closedtrades > 60 ? color.blue : color.gray) 
// *************************************************************************************************************************************************************************************************************************************************************************
// PLOTS
// *************************************************************************************************************************************************************************************************************************************************************************
// * strategy 1: MACD
// *************************************************************************************************************************************************************************************************************************************************************************
plot(trend, 'Trend', style=plot.style_columns, color=MACD > signal ? color.teal : color.gray, display=display.none, transp=30)
plot(MACD, 'MACD', color=color.new(color.blue, 0), display=display.none)
plot(signal, 'Signal', color=color.new(color.orange, 0), display=display.none)
plotshape(long_signal1 and show_signals ? up : na, 'Buy MACD', text='MACD', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0), display=display.none)
plotshape(short_signal1 and show_signals ? dn : na, 'Sell MACD', text='MACD', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0), display=display.none)
// *************************************************************************************************************************************************************************************************************************************************************************
// * strategy 2: Stoch RSI
// *************************************************************************************************************************************************************************************************************************************************************************
plotshape(long_signal2 and show_signals ? up : na, title='Buy Stoch RSI', text='SRSI', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0), display=display.none)
plotshape(short_signal2 and show_signals ? dn : na, title='Sell Stoch RSI', text='SRSI', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0), display=display.none)
// *************************************************************************************************************************************************************************************************************************************************************************
// * strategy 3: RSI
// *************************************************************************************************************************************************************************************************************************************************************************
plotshape(long_signal3 and show_signals ? up : na, title='Buy RSI', text='RSI', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0), display=display.none)
plotshape(short_signal3 and show_signals ? dn : na, title='Sell RSI', text='RSI', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0), display=display.none)
// *************************************************************************************************************************************************************************************************************************************************************************
// * strategy 4: Supertrend
// *************************************************************************************************************************************************************************************************************************************************************************
plotshape(long_signal4 and show_signals ? up : na, title='Buy Supertrend', text='Supertrend', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0), display=display.none)
plotshape(short_signal4 and show_signals ? dn : na, title='Sell Supertrend', text='Supertrend', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0), display=display.none)
// *************************************************************************************************************************************************************************************************************************************************************************
// * strategy 5: MA CROSS
// *************************************************************************************************************************************************************************************************************************************************************************
plotshape(long_signal5 and show_signals ? up : na, title='Buy MA CROSS', text='MA CROSS', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0), display=display.none)
plotshape(short_signal5 and show_signals ? dn : na, title='Sell MA CROSS', text='MA CROSS', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0), display=display.none)
// *************************************************************************************************************************************************************************************************************************************************************************
// * STRATEGY 6: POTENTIAL TOP/BOTTOM
// *************************************************************************************************************************************************************************************************************************************************************************
plotshape(condition_OB_several ? dn : na, title='Top', text='T', location=location.abovebar, style=shape.labeldown, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0))
plotshape(condition_OS_several ? up : na, title='Bottom', text='B', location=location.belowbar, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0))
// *************************************************************************************************************************************************************************************************************************************************************************
// * Buy/Sell signals
// *************************************************************************************************************************************************************************************************************************************************************************
w_total_long = weight_total(long_signal1, long_signal2, long_signal3, long_signal4, long_signal5)
w_total_short = weight_total(short_signal1, short_signal2, short_signal3, short_signal4, short_signal5)
plotshape(w_total_long >= weight_trigger and show_signals ? up : na, title='Buy Weigthed strategy', text='Buy', location=location.absolute, style=shape.labelup, size=size.tiny, color=color.new(color.teal, 0), textcolor=color.new(color.white, 0))
plotshape(w_total_short >= weight_trigger and show_signals ? dn : na, title='Sell Weigthed strategy', text='Sell', location=location.absolute, style=shape.labeldown, size=size.tiny, color=color.new(color.gray, 0), textcolor=color.new(color.white, 0))
// *************************************************************************************************************************************************************************************************************************************************************************
// * Stop loss targets
// *************************************************************************************************************************************************************************************************************************************************************************
plot(series=(strategy.position_size > 0) ? price_stop_long : na, color=color.gray, style=plot.style_cross, linewidth=2, transp=30, title="Long Stop Loss")
plot(series=(strategy.position_size < 0) ? price_stop_short : na, color=color.gray, style=plot.style_cross, linewidth=2, transp=30, title="Short Stop Loss")
// *************************************************************************************************************************************************************************************************************************************************************************
// * TP targets
// *************************************************************************************************************************************************************************************************************************************************************************
plot(strategy.position_size > 0 or strategy.position_size < 0 ? nextTP : na, color=color.aqua, style=plot.style_cross, linewidth=2, transp=30, title="Next TP")
// *************************************************************************************************************************************************************************************************************************************************************************
// * All strategies
// *************************************************************************************************************************************************************************************************************************************************************************
mPlot = plot(ohlc4, title='Price ohlc4', style=plot.style_circles, linewidth=0, display=display.none)
upPlot = plot((long_signal1 or long_signal2 or long_signal3 or long_signal4 or long_signal5) and (w_total_long > w_total_short) ? up : na, title='Up Trend', style=plot.style_linebr, linewidth=2, color=color.new(color.aqua, 0), display=display.none)
dnPlot = plot((short_signal1 or short_signal2 or short_signal3 or short_signal4 or short_signal5) and (w_total_short > w_total_long) ? dn : na, title='Down Trend', style=plot.style_linebr, linewidth=2, color=color.new(color.gray, 0), display=display.none)
plotchar(weight_trigger, "Trigger strategies", "", location.top, size = size.tiny, color=color.new(color.orange, 0))
plotchar(w_total_long, "Satisfied Long strategies", "", location.top, size = size.tiny, color=w_total_long >= weight_trigger ? color.orange : color.gray)
plotchar(w_total_short, "Satisfied Short strategies", "", location.top, size = size.tiny, color=w_total_long >= weight_trigger ? color.orange : color.gray)
plotshape((long_signal1 or long_signal2 or long_signal3 or long_signal4 or long_signal5) and (w_total_long > w_total_short) ? up : na, title='UpTrend Begins', location=location.absolute, style=shape.circle, size=size.tiny, color=color.new(color.aqua, 0), display=display.none)
plotshape((short_signal1 or short_signal2 or short_signal3 or short_signal4 or short_signal5) and (w_total_short > w_total_long) ? dn : na, title='DownTrend Begins', location=location.absolute, style=shape.circle, size=size.tiny, color=color.new(color.gray, 0), display=display.none)
fill(mPlot, upPlot, title='UpTrend Highligter', color=colors('buy', 80))
fill(mPlot, dnPlot, title='DownTrend Highligter', color=colors('sell', 80))

// *************************************************************************************************************************************************************************************************************************************************************************
// MONTHLY TABLE PERFORMANCE - Developed by @QuantNomad
// *************************************************************************************************************************************************************************************************************************************************************************
show_performance = input.bool(true, 'Show Monthly Performance ?', group='Performance - credits: @QuantNomad')
prec = input(2, 'Return Precision', group='Performance - credits: @QuantNomad')

if show_performance
    new_month = month(time) != month(time[1])
    new_year  = year(time)  != year(time[1])
    
    eq = strategy.equity
    
    bar_pnl = eq / eq[1] - 1
    
    cur_month_pnl = 0.0
    cur_year_pnl  = 0.0
    
    // Current Monthly P&L
    cur_month_pnl := new_month ? 0.0 : 
                     (1 + cur_month_pnl[1]) * (1 + bar_pnl) - 1 
    
    // Current Yearly P&L
    cur_year_pnl := new_year ? 0.0 : 
                     (1 + cur_year_pnl[1]) * (1 + bar_pnl) - 1  
    
    // Arrays to store Yearly and Monthly P&Ls
    var month_pnl  = array.new_float(0)
    var month_time = array.new_int(0)
    
    var year_pnl  = array.new_float(0)
    var year_time = array.new_int(0)
    
    last_computed = false
    
    if (not na(cur_month_pnl[1]) and (new_month or barstate.islastconfirmedhistory))
        if (last_computed[1])
            array.pop(month_pnl)
            array.pop(month_time)
            
        array.push(month_pnl , cur_month_pnl[1])
        array.push(month_time, time[1])
    
    if (not na(cur_year_pnl[1]) and (new_year or barstate.islastconfirmedhistory))
        if (last_computed[1])
            array.pop(year_pnl)
            array.pop(year_time)
            
        array.push(year_pnl , cur_year_pnl[1])
        array.push(year_time, time[1])
    
    last_computed := barstate.islastconfirmedhistory ? true : nz(last_computed[1])
    
    // Monthly P&L Table    
    var monthly_table = table(na)
    
    if (barstate.islastconfirmedhistory)
        monthly_table := table.new(position.bottom_right, columns = 14, rows = array.size(year_pnl) + 1, border_width = 1)
    
        table.cell(monthly_table, 0,  0, "",     bgcolor = #cccccc)
        table.cell(monthly_table, 1,  0, "Jan",  bgcolor = #cccccc)
        table.cell(monthly_table, 2,  0, "Feb",  bgcolor = #cccccc)
        table.cell(monthly_table, 3,  0, "Mar",  bgcolor = #cccccc)
        table.cell(monthly_table, 4,  0, "Apr",  bgcolor = #cccccc)
        table.cell(monthly_table, 5,  0, "May",  bgcolor = #cccccc)
        table.cell(monthly_table, 6,  0, "Jun",  bgcolor = #cccccc)
        table.cell(monthly_table, 7,  0, "Jul",  bgcolor = #cccccc)
        table.cell(monthly_table, 8,  0, "Aug",  bgcolor = #cccccc)
        table.cell(monthly_table, 9,  0, "Sep",  bgcolor = #cccccc)
        table.cell(monthly_table, 10, 0, "Oct",  bgcolor = #cccccc)
        table.cell(monthly_table, 11, 0, "Nov",  bgcolor = #cccccc)
        table.cell(monthly_table, 12, 0, "Dec",  bgcolor = #cccccc)
        table.cell(monthly_table, 13, 0, "Year", bgcolor = #999999)
    
        for yi = 0 to array.size(year_pnl) - 1
            table.cell(monthly_table, 0,  yi + 1, str.tostring(year(array.get(year_time, yi))), bgcolor = #cccccc)
            
            y_color = array.get(year_pnl, yi) > 0 ? color.new(color.teal, transp = 40) : color.new(color.gray, transp = 40)
            table.cell(monthly_table, 13, yi + 1, str.tostring(math.round(array.get(year_pnl, yi) * 100, prec)), bgcolor = y_color, text_color=color.new(color.white, 0))
            
        for mi = 0 to array.size(month_time) - 1
            m_row   = year(array.get(month_time, mi))  - year(array.get(year_time, 0)) + 1
            m_col   = month(array.get(month_time, mi)) 
            m_color = array.get(month_pnl, mi) > 0 ? color.new(color.teal, transp = 40) : color.new(color.gray, transp = 40)
            
            table.cell(monthly_table, m_col, m_row, str.tostring(math.round(array.get(month_pnl, mi) * 100, prec)), bgcolor = m_color, text_color=color.new(color.white, 0))
            
plot(ext_signal, 'External Signal', display=display.none)