import openai
import prompt_toolkit
import time

OPENAI_API_KEY = 'dqss'
openai.api_key = OPENAI_API_KEY



# message=[{'role': 'user', 'content': 'ELI5 the concept of sublimation in physics.'}]
# response = openai.ChatCompletion.create(
# model="gpt-3.5-turbo",
# max_tokens=4000,
# temperature=1.2,
# messages = message)


# print(response['choices'][0]['message']['content'])

# typing machine
# for choice in response.choices:
#     for character in choice.message.content:
#         print(character, end='', flush=True)
#         time.sleep(0.05)
#     print()


# Define a function to prompt the user for input and return the OpenAI response
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.2,
    )

    return response['choices'][0]['message']['content']

# Set up the prompt toolkit interface
def main():
    while True:
        # Prompt the user for input
        user_input = prompt_toolkit.prompt("You: ")
        message=[{"role": "system", "content": "Conversation scientifique et complète."},                 
                 {'role': 'user', 'content': user_input}
                 ]
        # Generate a response using OpenAI
        openai_response = get_openai_response(message)

        # Print the response
        print(openai_response)

if __name__ == '__main__':
    main()

