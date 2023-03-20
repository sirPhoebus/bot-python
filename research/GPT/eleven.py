import openai
import requests
import json
import pygame
from io import BytesIO

OPENAI_API_KEY = 'magic'
openai.api_key = OPENAI_API_KEY

message = [{'role': 'user', 'content': 'tell me a joke'}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=4000,
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
    'xi-api-key': 'magic'
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
    print(audio_data)
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
