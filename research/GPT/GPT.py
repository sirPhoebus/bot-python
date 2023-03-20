import openai

import time

OPENAI_API_KEY = 'magic key'
openai.api_key = OPENAI_API_KEY



message=[{'role': 'user', 'content': 'ELI5 the concept of sublimation in physics.'}]
response = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
max_tokens=4000,
temperature=1.2,
messages = message)


print(response['choices'][0]['message']['content'])

# typing machine
# for choice in response.choices:
#     for character in choice.message.content:
#         print(character, end='', flush=True)
#         time.sleep(0.05)
#     print()

