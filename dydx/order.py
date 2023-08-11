import requests
import json
import time
import hmac
import hashlib

def place_order(api_key, api_secret, market, side, size, price):
    # Endpoint URL
    url = 'https://api.dydx.exchange/v3/orders'

    # Timestamp
    timestamp = str(int(time.time() * 1000))

    # Order details
    order = {
        'market': market,
        'side': side,
        'size': size,
        'price': price,
        'postOnly': True,
        'cancelAfter': 'day'
    }

    # Create the request body
    body = json.dumps(order)

    # Create the signature
    signature = hmac.new(
        api_secret.encode('utf-8'),
        (timestamp + body).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Headers
    headers = {
        'DYDX-API-KEY': api_key,
        'DYDX-API-PASSPHRASE': api_secret,  # If required
        'DYDX-API-TIMESTAMP': timestamp,
        'DYDX-API-SIGNATURE': signature,
        'Content-Type': 'application/json'
    }

    # Send the request
    response = requests.post(url, data=body, headers=headers)

    # Return the response
    return response.json()
