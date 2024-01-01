import socketio


class Socket:

    def __init__(self):
        self.__sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.__asgi = socketio.ASGIApp(self.__sio)

    async def send_data(self, channel: str, data: dict):
        print(data)
        await self.__sio.emit(channel, data)
        return

    def __call__(self):
        return self.__asgi


import time
import socketio

class SocketIOClient:
    def __init__(self, server_url, attemp=5):
        self.sio = socketio.Client()

        # Define event handlers
        self.sio.on("connect", self.handle_connect)
        self.sio.on("message", self.handle_message)
        self.sio.on("disconnect", self.handle_disconnect)
        self._server_url = server_url
        self._attemp = attemp
        # Connect to the server
        self.connect(server_url)

    def connect(self, server_url):
        try:
            self.sio.connect(server_url)
        except Exception as e:
            print(f"Failed to connect to the server: {e}")
            self.retry_connection()

    def retry_connection(self):
        # Retry the connection every 5 seconds
        while not self.sio.connected and self._attemp > 0:
            print("Retrying connection in 5 seconds...")
            time.sleep(5)
            try:
                self.sio.connect(self._server_url)
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

    def send_alarm(self, data):
        self.sio.send(data)
    
    def disconnect(self):
        self.sio.disconnect()


socket_connection = Socket()
