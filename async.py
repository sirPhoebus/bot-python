import asyncio
import json
import pandas as pd
import websockets
import unicorn_binance_websocket_api

async def process_message(msg):
    data = json.loads(msg)
    k_dict = data['data']['k']

    try:
        df = pd.DataFrame.from_dict(k_dict, orient='index').transpose()
        return df

    except ValueError:
        print("Error: The shape of the data is different from the dataframe.")

async def binance_server(websocket, path):
    ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
    ubwa.create_stream(['kline_1m'], ['btcusdt'])
    skip_first_message = True
    while True:
        oldest_data_from_stream_buffer = ubwa.pop_stream_data_from_stream_buffer()
        if oldest_data_from_stream_buffer:
            if skip_first_message:
                skip_first_message = False
                continue
            df = await process_message(oldest_data_from_stream_buffer)
            print(df)
            await websocket.send(json.dumps(df.to_dict()))
            await asyncio.sleep(1)  # to avoid blocking

start_server = websockets.serve(binance_server, "localhost", 8888)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
