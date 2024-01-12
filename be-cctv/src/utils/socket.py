import socketio
import asyncio
import time
from typing import Dict, Any
import base64
import io

class Socket:

    def __init__(self):
        self.__sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.__asgi = socketio.ASGIApp(self.__sio)

    async def send_data(self, channel: str, data: dict):
        await self.__sio.emit(channel, data)
        return

    def __call__(self):
        return self.__asgi


class SocketIOClient:
    def __init__(self, server_url, attemp=5):
        self.sio = socketio.Client()

        # Define event handlers
        self.sio.on("connect", self.handle_connect)
        self.sio.on("disconnect", self.handle_disconnect)
        self._server_url = server_url
        self._attemp = attemp
        # Connect to the server
        self.connect(server_url)

    def connect(self, server_url):
        try:
            self.sio.connect(server_url, socketio_path='/ws/vms')

        except Exception as e:
            print(f"Failed to connect to the server: {e}")
            self.retry_connection()

    def retry_connection(self):
        # Retry the connection every 5 seconds
        while not self.sio.connected and self._attemp > 0:
            print("Retrying connection in 5 seconds...")
            time.sleep(5)
            try:
                self.sio.connect(self._server_url,  socketio_path='/ws/vms')
            except Exception as e:
                print(f"Failed to connect to the server: {e}")
                self._attemp -= 1

    def handle_connect(self):
        print("Connected to server")

    def handle_message(self, data):
        print(f"Message from server: {data}")

    def handle_disconnect(self):
        print("Disconnected from server")
        self.retry_connection()

    def send_message(self, message):
        self.sio.send(message)

    def send_alarm(self, channel, data):
        self.sio.emit(channel, data)
        
    def send_lp(self, channel, data):
        self.sio.emit(channel, data)

    
    def disconnect(self):
        self.sio.disconnect()

class AsyncSocketIOClient:
    def __init__(self, server_url: str, socketio_path: str = "/ws/vms"):
        self.server_url = server_url
        self.socketio_path = socketio_path
        self.sio = socketio.AsyncClient()

    async def connect(self):
        await self.sio.connect(self.server_url, socketio_path=self.socketio_path)
        print(f"Connected to {self.server_url} with path {self.socketio_path}")

    async def disconnect(self):
        await self.sio.disconnect()
        print("Disconnected from socket")
    
    async def healthy_check_event(self, channel: str, data: Dict[str, Any]):
        await self.sio.emit(channel, data)
        
    async def emit_event(self, channel: str, data: Dict[str, Any]):
        print(data['lp'])
        await self.sio.emit(channel, data)


socket_connection = Socket()
