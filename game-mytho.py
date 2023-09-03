
import random, json
import requests
from randomwordfr import RandomWordFr
from twilio.rest import Client
ACCOUNT_SID = 'xxx'
AUTH_TOKEN = 'xxx'

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ID	Catégories	Endpoints
# 1	L'ÉCOLE – LA CLASSE – L'INSTRUCTION	
# "https://trouve-mot.fr/api/categorie/1/7
# 2	PAYSAGES – CLIMAT – FORMES	
# "https://trouve-mot.fr/api/categorie/2/5
# 3	QUALITÉS ET DÉFAUTS	
# "https://trouve-mot.fr/api/categorie/3/2
# 4	CALCUL ET MESURES	
# "https://trouve-mot.fr/api/categorie/4/5
# 5	LES ALIMENTS – LES BOISSONS- LES REPAS	
# "https://trouve-mot.fr/api/categorie/5/3
# 6	LE CORPS HUMAIN	
# "https://trouve-mot.fr/api/categorie/6/2
# 7	LES SENS – LA VOLONTE – L'INTELLIGENCE	
# "https://trouve-mot.fr/api/categorie/7/1
# 8	L'INTÉRIEUR ET LE MOBILIER	
# "https://trouve-mot.fr/api/categorie/8/1
# 9	L'INDUSTRIE ET LE TRAVAIL	
# "https://trouve-mot.fr/api/categorie/9/8
# 10	LES ARTS	
# "https://trouve-mot.fr/api/categorie/10/6
# 11	L'AGRICULTURE	
# "https://trouve-mot.fr/api/categorie/11/8
# 12	VERGER – BOIS – CHASSE - PÊCHE	
# "https://trouve-mot.fr/api/categorie/12/5
# 13	GESTES ET MOUVEMENTS	
# "https://trouve-mot.fr/api/categorie/13/2
# 14	ÉPOQUE – TEMPS - SAISONS	
# "https://trouve-mot.fr/api/categorie/14/2
# 15	VÊTEMENTS – TOILETTE - PARURES	
# "https://trouve-mot.fr/api/categorie/15/8
# 16	SPORTS ET JEUX	
# "https://trouve-mot.fr/api/categorie/16/7
# 17	LA MAISON – LE BATIMENT	
# "https://trouve-mot.fr/api/categorie/17/5
# 18	LES VOYAGES	
# "https://trouve-mot.fr/api/categorie/18/1
# 19	LES ANIMAUX	
# "https://trouve-mot.fr/api/categorie/19/6
# 20	VILLE – VILLAGE – UNIVERS - DIMENSIONS	
# "https://trouve-mot.fr/api/categorie/20/4
# 21	EAUX – MINÉRAUX - VÉGÉTAUX	
# "https://trouve-mot.fr/api/categorie/21/8
# 22	LE COMMERCE	
# "https://trouve-mot.fr/api/categorie/22/1
# 23	LA COMMUNICATION	
# "https://trouve-mot.fr/api/categorie/23/3
# 24	JOIES ET PEINES	
# "https://trouve-mot.fr/api/categorie/24/1
# 25	GOUVERNEMENT ET JUSTICE	
# "https://trouve-mot.fr/api/categorie/25/3
# 26	L'ARMÉE	
# "https://trouve-mot.fr/api/categorie/26/4
# 27	VIE HUMAINE – MALADIES - HYGIÈNE	
# "https://trouve-mot.fr/api/categorie/27/8

# Very complex words
def generate():
    url = "https://trouve-mot.fr/api/categorie/6/2"

    headers = {

    }

    response = requests.get(url, headers=headers)

    return response.json()
    
#Very complex words >> Dictionaire 
def create():
    words = []
    for i in range(2):
        rw = RandomWordFr()
        r = rw.get()
        w = r['word']
        print(w)
        words.append(w)
    return words 
    
def get_player_numbers(nbr_of_players):
    """Asks for the mobile number of each player"""
    player_numbers = []
    
    for i in range(nbr_of_players):
        while True:
            try:
                num = str(input("Enter your mobile number: "))
                player_numbers.append(num)
                break
            except ValueError:
                print("Invalid input. Please enter only digits.")
    
    return player_numbers

# def get_random_words():
#     # OpenAI endpoint for the Completion API (as of 2021)
#     url = "https://api.openai.com/v1/engines/davinci/completions"
#     headers = {
#         'Authorization': "Bearer TOKEN",
#         'Content-Type': 'application/json'
#     }
#     data = {
#         'prompt': 'Ecrire deux substantifs qui sont matériels.',
#         'max_tokens': 5
#     }

#     try:
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()
#         api_response = response.json()
        
#         words = api_response['choices'][0]['text'].strip().split()
#         if len(words) < 2:
#             return ("word1", "word2")  # Default values in case the API doesn't provide enough words
#         else:
#             return (words[0], words[1])

#     except requests.RequestException:
#         print("Failed to fetch random words from the API.")
#         return ("word1", "word2")  # Default values

# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run():
    request = {
        'prompt': '###Instruction: Write two random words which are separated by a comma. ###Response:',
        'max_new_tokens': 10,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 1.99,
        'top_p': 0.9,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.15,
        'repetition_penalty_range': 0,
        'top_k': 20,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 10,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        words = result.split()
        return words
def send_msg(word, player):
    """Sends the message to the player"""
    print(f"sending :  {word} to {player}")
    client.messages.create(
    body= word, 
    from_='+16562184220',
    to= player)

def distribute_words(player_numbers, common_words):
    """Distributes the common words to the players"""
    random.shuffle(player_numbers)
    random.shuffle(common_words)  # Shuffle common words
    
    # Distribute the first word to all players except the last one
    for i in range(len(player_numbers)-1):
        send_msg(common_words[0], player_numbers[i])
        
    # Send the second word to the last player
    send_msg(common_words[1], player_numbers[-1])

if __name__ == '__main__':
    nbr_of_players = int(input("How many players are playing? "))
    player_numbers = get_player_numbers(nbr_of_players)
    
    while True:
        data = generate()
        #print(data)
        words = [item["name"] for item in data]
        distribute_words(player_numbers, words)
        
        # Ask the user if they want to start a new round
        continue_game = input("Do you want to start a new round? (yes/no): ").strip().lower()
        if continue_game != 'yes':
            break
