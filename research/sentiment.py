import numpy as np
from pycoingecko import CoinGeckoAPI
import requests
import snscrape.modules.twitter as sntwitter
import pandas as pd
import textwrap
import datetime as dt
import openai

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

openai.api_key = "YOUR KEY"

symbols = ["BTC", "ETH"]

def get_all_crypto_news():
    API_KEY = "ff795da3afca47fe9ef344bc08e7d9d2"
    all_news = {}

    for symbol in symbols:
        url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}'
        response = requests.get(url)
        data = response.json()

        news_data = []
        try:
            for article in data['articles'][:10]:  # Limit to top 3 articles
                news_data.append({
                    'title': article['title'],
                    'source': article['source']['name'],
                })
            all_news[symbol] = news_data
        except:
            return all_news

    return all_news

def getTweets():
    queries = ["BTC", "ETH"]
    tweets_list = []

    for query in queries:
        num = 0
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if num == 10:
                break
            num += 1
            tweets_list.append(tweet.rawContent)
            return tweet.rawContent
        
base_prompt = f"""
You are in control of my crypto trading profile. You should take into consideration the factors you have to determine the best trade. Here is the info:

You can execute these commands:

1. buy_crypto_price(symbol, amount)
2. buy_crypto_limit(symbol, amount, limit)
3. sell_crypto_price(symbol, amount)
4. sell_crypto_limit(symbol, amount, limit)
5. do_nothing()

Use this when you don't see any necessary changes.

You also have access to this data:

1. Historical data
2. News Headlines
3. Twitter Data
4. Vector Delta

Vector delta measures how similar the last iterations market environment was to this one. it's a number between 0 and 1, where 1 is the most similar.

The current date and time is {dt.datetime.today()}

You are called once every 30 minutes, keep this in mind.

The only cryptos you can trade are LINK, ETH, MATIC, SOL and LTC.

here are the data sources:


"""

# Convert the info into a format suitable for the AI prompt
#info_str = f"Historical data: {crypto_data}\n News: {news_output} Twitter Data: {tweets_list}\n Vector Delta: {vector_delta}"
#prompt = base_prompt + "\n\n" + info_str
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
def getDecision():
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
    return textwrap.fill(str(res), width=50)


# Download necessary NLTK datasets and tokenizers
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Function to create a feature dictionary of all words in a list
def create_word_features(words):
    return dict([(word, True) for word in words])

# Load movie reviews from NLTK
positive_ids = movie_reviews.fileids('pos')
negative_ids = movie_reviews.fileids('neg')

# Create a list of tuples with words from reviews and their labels
positive_features = [(create_word_features(movie_reviews.words(fileids=[f])), 'positive') for f in positive_ids]
negative_features = [(create_word_features(movie_reviews.words(fileids=[f])), 'negative') for f in negative_ids]

# Combine positive and negative features
features = positive_features + negative_features

# Split into training and testing datasets
train_set, test_set = features[200:], features[:200]

# Train a Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the accuracy of the classifier on the test set
print(f"Classifier accuracy: {accuracy(classifier, test_set)}")

def calculate_global_sentiment(sentiment_list):
    """
    Calculate the global average sentiment score from a list of sentiment dictionaries.

    Parameters:
    sentiment_list (list of dict): A list where each element is a sentiment dictionary
                                   with keys 'neg', 'neu', 'pos', and 'compound'.

    Returns:
    float: The average compound sentiment score.
    """
    # Sum all the compound scores
    total_compound = sum(sentiment['compound'] for sentiment in sentiment_list)
    
    # Divide by the number of entries
    average_sentiment = total_compound / len(sentiment_list) if sentiment_list else 0
    
    return average_sentiment


def analyze_sentiment(text):
    # Ensure text is a string here. If it's not, you need to debug why.
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, but got a {type(text)}")

    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)




news_output = get_all_crypto_news()
#print(textwrap.fill(str(news_output), width=50))
#print(news_output)

for currency, articles in news_output.items():
    sentiments = []
    for article in articles:
        title = article['title']
        sentiment = analyze_sentiment(title)  # Ensure that analyze_sentiment accepts and processes a string.
        sentiments.append(sentiment)
        print(f"Sentiment of '{title}': {sentiment}")

    # Now calculate the global sentiment for the current currency
    global_sentiment = calculate_global_sentiment(sentiments)
    print(f"Global Sentiment for {currency}: {global_sentiment}")

    # Range: The compound score ranges from -1 to +1.
    # Negative Sentiment: Scores closer to -1 indicate a more negative sentiment.
    # Positive Sentiment: Scores closer to +1 indicate a more positive sentiment.
    # Neutral Sentiment: Scores around 0 indicate a neutral sentiment or a balanced amount of positive and negative sentiments.