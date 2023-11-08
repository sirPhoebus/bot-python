import json
import httpx
from parsel import Selector
import re
# Define the headers to simulate a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.54 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

# Replace 'scrapfly_dev' with the target Twitter handle
URL = "https://syndication.twitter.com/srv/timeline-profile/screen-name/CryptoBoomNews"

# Make the GET request to retrieve the embed HTML
with httpx.Client(http2=True, headers=HEADERS) as client:
    response = client.get(URL)
    assert response.status_code == 200

# Parse the response text with Selector
sel = Selector(response.text)

# Find the JSON data embedded in the HTML and parse it
data = json.loads(sel.css("script#__NEXT_DATA__::text").get())

# Extract tweet information from the JSON data
tweet_data = data["props"]["pageProps"]["timeline"]["entries"]
tweets = [tweet["content"]["tweet"] for tweet in tweet_data]
# Extract pure text from tweets
pure_text_tweets = [tweet["content"]["tweet"]["full_text"] for tweet in tweet_data if "full_text" in tweet["content"]["tweet"]]

# Define a regular expression pattern to match URLs
url_pattern = r'https?://\S+'

# Remove URLs from each tweet's text
tweets_without_links = [re.sub(url_pattern, '', tweet_text) for tweet_text in pure_text_tweets]

# Print the tweets
#print(tweets)
# for text in pure_text_tweets:
#     print(text)
# Print the text of each tweet without any links
for text in tweets_without_links:
    print(text)