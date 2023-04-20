import pandas as pd
from datetime import datetime

def match_orders(orders):
    """
    Matches buy and sell orders and returns a list of trades.
    
    Args:
        orders: a list of dictionaries representing buy and sell orders.
                Each order dictionary should have the following keys:
                - 'order_id': a unique identifier for the order
                - 'timestamp': a string representing the time the order was placed
                - 'side': either 'buy' or 'sell'
                - 'price': the price at which the order is placed
                - 'quantity': the quantity of the asset being traded
    
    Returns:
        A list of dictionaries representing trades. Each trade dictionary should have the following keys:
        - 'timestamp': a string representing the time the trade was executed
        - 'buy_order_id': the order ID of the buy order
        - 'sell_order_id': the order ID of the sell order
        - 'price': the price at which the trade was executed
        - 'quantity': the quantity of the asset that was traded
    """
    
    # Load orders into a Pandas DataFrame
    orders_df = pd.DataFrame(orders)
    
    # Convert timestamp strings to datetime objects
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    
    # Sort orders by price and date (oldest first)
    orders_df = orders_df.sort_values(by=['price', 'timestamp'], ascending=[True, True])
    
    # Initialize variables to keep track of the current buy and sell orders
    current_buy_order = None
    current_sell_order = None
    
    # Initialize a list to keep track of trades
    trades = []
    
    # Loop through the orders and match buy and sell orders
    for _, order in orders_df.iterrows():
        if order['side'] == 'buy':
            # If a buy order is encountered, update the current buy order
            current_buy_order = order
        elif order['side'] == 'sell':
            # If a sell order is encountered, update the current sell order
            current_sell_order = order
        
        # If there are both a current buy and sell order, execute a trade
        if current_buy_order is not None and current_sell_order is not None:
            trade_quantity = min(current_buy_order['quantity'], current_sell_order['quantity'])
            trade_price = current_sell_order['price']
            
            # Create a trade dictionary and add it to the trades list
            trade = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'buy_order_id': current_buy_order['order_id'],
                'sell_order_id': current_sell_order['order_id'],
                'price': trade_price,
                'quantity': trade_quantity
            }
            trades.append(trade)
            
            # Update the quantity of the buy and sell orders
            current_buy_order['quantity'] -= trade_quantity
            current_sell_order['quantity'] -= trade_quantity
            
            # If the buy order has been completely filled, set the current buy order to None
            if current_buy_order['quantity'] == 0:
                current_buy_order = None
            
            # If the sell order has been completely filled, set the current sell order to None
            if current_sell_order['quantity'] == 0:
                current_sell_order = None
    
    return trades


orders = []

# Add a buy order to the order book
buy_order = {
    'order_id': 1,
    'timestamp': '2023-04-14 10:00:00',
    'side': 'buy',
    'price': 50000,
    'quantity': 10
    }
orders.append(buy_order)

sell_order = {
'order_id': 2,
'timestamp': '2023-04-14 10:01:00',
'side': 'sell',
'price': 51000,
'quantity': 5
}
orders.append(sell_order)

trades = match_orders(orders)
print(trades)

buy_order_2 = {
'order_id': 3,
'timestamp': '2023-04-14 10:02:00',
'side': 'buy',
'price': 49000,
'quantity': 10
}
orders.append(buy_order_2)

orders = match_orders(orders)

Add another sell order to the order book
sell_order_2 = {
'order_id': 4,
'timestamp': '2023-04-14 10:03:00',
'side': 'sell',
'price': 52000,
'quantity': 5
}
orders.append(sell_order_2)

trades = match_orders(orders)
print(trades)