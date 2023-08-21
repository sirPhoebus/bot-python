import tornado.web
import tornado.websocket
import json
import pandas as pd
import unicorn_binance_websocket_api
import asyncio

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def get(self):
        self.stop = False
        self.ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com")
        self.ubwa.create_stream(['kline_1m'], ['btcusdt'])
        asyncio.ensure_future(self.write_data())
        
    async def write_data(self):
        while not self.stop:
            oldest_data_from_stream_buffer = self.ubwa.pop_stream_data_from_stream_buffer()
            self.write(json.dumps(oldest_data_from_stream_buffer))
            await asyncio.sleep(1)  # waiting for 1 sec to avoid blocking

    def on_close(self):
        self.stop = True
        self.ubwa.stop_manager()

def make_app():
    return tornado.web.Application([
        (r"/", WebSocketHandler),
    ])

async def main():
    app = make_app()
    app.listen(8888)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())