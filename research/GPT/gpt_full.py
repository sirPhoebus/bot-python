import openai
import prompt_toolkit
from prompt_toolkit import PromptSession
import datetime

# Define the filename for the text file
filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"



OPENAI_API_KEY = 'dd'
openai.api_key = OPENAI_API_KEY

conversation = []

# Define a function to prompt the user for input and return the OpenAI response
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=1.2,
    )
    return response['choices'][0]['message']['content']
# Set up the prompt toolkit interface
def main():
    while True:
        # Prompt the user for input
        user_input = prompt_toolkit.prompt("You: ")
        message = {'role': 'user', 'content': user_input}
        conversation.append(message)

        openai_response = get_openai_response(conversation)

        # Store the response and print it
        message = {'role': 'assistant', 'content': openai_response}
        conversation.append(message)
        print("Bot: " + openai_response)
        # Open the file in write mode
        with open(filename, "w") as f:
            # Loop through each message in the conversation
            for message in conversation:
                # Write the message to the file
                f.write(message['role'] + ": " + message['content'] + "\n")
                
if __name__ == '__main__':
    main()
