import requests
from bs4 import BeautifulSoup
import os

# URL of the gallery page
url = 'https://onlyfans.com/paigebritish/media'

# Make a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the elements that contain the links to the individual images in the gallery
image_links = soup.find_all('a', class_='image-link')

# Create a directory to save the images in
if not os.path.exists('images'):
    os.makedirs('images')

# Download each image and save it locally
for link in image_links:
    image_url = link['href']
    filename = image_url.split('/')[-1]
    response = requests.get(image_url)
    with open('images/' + filename, 'wb') as f:
        f.write(response.content)