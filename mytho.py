import asyncio
import websockets
import random
from randomwordfr import RandomWordFr
import requests
connected_clients = set()


# Very complex words
def generate():
    url = "https://trouve-mot.fr/api/categorie/6/2"

    headers = {

    }

    response = requests.get(url, headers=headers)

    return response.json()
    
# Assuming RandomWordFr is defined somewhere else
def create():
    words = []
    for i in range(2):
        rw = RandomWordFr()
        r = rw.get()
        w = r['word']
        #print(w)
        words.append(w)
    return words 

connected_clients = set()

async def send_word(websocket, path):
    connected_clients.add(websocket)
    try:
        while True:
            words = create()  # Get 2 new random words
            special_client = random.choice(list(connected_clients))  # Select a random client

            try:
                # Send the first word to the chosen client
                await special_client.send(words[0])
            except websockets.ConnectionClosedError:
                connected_clients.remove(special_client)

            disconnected_clients = set()
            for client in connected_clients - {special_client}:  # Send the second word to all other clients
                try:
                    await client.send(words[1])
                except websockets.ConnectionClosed:
                    disconnected_clients.add(client)
                except websockets.ConnectionClosedError:  # Handle the case where the connection is already closed
                    disconnected_clients.add(client)
            
            # Remove disconnected clients from the set
            connected_clients.difference_update(disconnected_clients)
            print(f"Nbr of clients: {connected_clients}")
            await asyncio.sleep(10)
    finally:
        connected_clients.remove(websocket)

async def start():
    server = await websockets.serve(send_word, "localhost", 8765)
    await server.wait_closed()

asyncio.run(start())
