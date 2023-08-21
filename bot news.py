import numpy as np

import requests
import pandas as pd
import textwrap
import datetime as dt
import openai

symbols = ["LINK", "ETH", "PRE", "SOL", "PRE"]

def get_all_crypto_news():
    API_KEY = "343102c3541a4bb7b189be0cae6997f0"
    all_news = {}

    for symbol in symbols:
        url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}'
        response = requests.get(url)
        data = response.json()

        news_data = []
        try:
            for article in data['articles'][:3]:  # Limit to top 3 articles
                news_data.append({
                    'title': article['title'],
                    'source': article['source']['name'],
                })
            all_news[symbol] = news_data
        except:
            return all_news

    return all_news

news_output = get_all_crypto_news()
print(textwrap.fill(str(news_output), width=50))


        
base_prompt = f"""
You are in control of my crypto trading profile. You should take into consideration the factors you have to determine the best trade. Here is the info:

You can execute these commands:

1. buy_crypto_price(symbol, amount)
2. buy_crypto_limit(symbol, amount, limit)
3. sell_crypto_price(symbol, amount)
4. sell_crypto_limit(symbol, amount, limit)
5. do_nothing()

You also have access to this data News Headlines

The current date and time is {dt.datetime.today()}

The only cryptos you can trade are {symbols}.

"""

# Convert the info into a format suitable for the AI prompt
info_str = f"News: {news_output}"
prompt = base_prompt + "\n\n" + info_str
user_prompt = """
What should we do to make the most amount of profit based on the info? Here are your options for a response.

1. buy_crypto_price(symbol, amount) This will buy the specified amount of the specified cryptocurrency.
2. buy_crypto_limit(symbol, amount, limit) This will set a limit order to buy the specified amount of the specified cryptocurrency if it reaches the specified limit.
3. sell_crypto_price(symbol, amount) This will sell the specified amount of the specified cryptocurrency.
4. sell_crypto_limit(symbol, amount, limit) This will set a limit order to sell the specified amount of the specified cryptocurrency if it reaches the specified limit.
5. do_nothing() Use this when you don't see any necessary changes.

Choose one.
CRITICAL: RESPOND IN ONLY THE ABOVE FORMAT. EXAMPLE: buy_crypto_price("ETHBTC", 0.1). DO NOT SAY ANYTHING ELSE.
ALSO IN THE AMOUNT FIELD, USE THE UNIT SYSTEM OF BITCOIN, NOT DOLLARS. ASSUME WE HAVE A BUDGET of UP TO $100 WORTH OF BITCOIN PER TRADE for 24 hours.
 ADD THE ACRONYM "BTC" AT THE END OF THE CRYPTO TICKER.
    """

openai.api_key = "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"

# Feed the prompt to the AI
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature = 0.2,
    )

res = response.choices[0].message["content"]
res = res.replace("\\", "")
print(textwrap.fill(str(res), width=50))