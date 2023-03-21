import openai
import requests
import json
import pygame
from io import BytesIO

OPENAI_API_KEY = 'key'
openai.api_key = OPENAI_API_KEY
eleven_key = 'key'
message = [{'role': 'user', 'content': 'What is the pros of using the pyhton interactive window in Visual Code compare to just running the code in the terminal ?'}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=2000,
    temperature=1.2,
    messages=message
)

# Extract the response message
response_text = response.choices[0].message['content']
print(response_text)

ELEVENLABS_API_URL = 'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM'
headers = {
    'accept': 'audio/mpeg',
    'Content-Type': 'application/json',
    'xi-api-key': eleven_key
}
data = {
    "text": response_text,
    "voice_settings": {
        "stability": 0.1,
        "similarity_boost": 0.1
    }
}
response_elevenlabs = requests.post(ELEVENLABS_API_URL, headers=headers, data=json.dumps(data))
if response_elevenlabs.status_code == 200:
    # Load the audio data and play it
    audio_data = BytesIO(response_elevenlabs.content)
# Initialize Pygame
pygame.init()

# Load the audio data into Pygame
pygame.mixer.music.load(audio_data)

# Play the audio
pygame.mixer.music.play()

# Wait until the audio has finished playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

# Clean up Pygame resources
pygame.quit()
