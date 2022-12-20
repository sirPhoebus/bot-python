import time
import requests

# API endpoint for getting the latest Bitcoin price
url = "https://api.coindesk.com/v1/bpi/currentprice/BTC.json"

# Initialize variables
prev_price = None
trend = None

while True:
    # Make a request to the API endpoint to get the latest Bitcoin price
    response = requests.get(url)
    data = response.json()

    # Extract the current price from the API response
    current_price = data["bpi"]["USD"]["rate_float"]

    # Check if the current price is higher or lower than the previous price
    if prev_price is not None:
        if current_price > prev_price:
            trend = "uptrend"
        elif current_price < prev_price:
            trend = "downtrend"
        else:
            trend = "neutral"

    # Print the current price and trend
    print(f"Current price: {current_price} ({trend})")

    # Update the previous price
    prev_price = current_price

    # Sleep for 1 minute
    time.sleep(60)
