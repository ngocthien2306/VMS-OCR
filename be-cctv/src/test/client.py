
import time
import socketio

class SocketIOClient:
    def __init__(self, server_url):
        self.sio = socketio.Client()

        # Define event handlers
        self.sio.on("connect", self.handle_connect)
        self.sio.on("message", self.handle_message)
        self.sio.on("disconnect", self.handle_disconnect)

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
        while not self.sio.connected:
            print("Retrying connection in 5 seconds...")
            time.sleep(5)
            try:
                self.sio.connect("http://dev.i-soft.com.vn:6000")
            except Exception as e:
                print(f"Failed to connect to the server: {e}")

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



if __name__ == "__main__":
    client = SocketIOClient("http://dev.i-soft.com.vn:6000")

    # Send a message to the server
    client.send_message("Hello, server!")

    # Wait for the user to press Enter before disconnecting
    start_time = time.time()
    while True:
        if time.time() - start_time > 1.0:
            try: 
                client.send_alarm({'camera_id': 'camera_1', 'status': 'alarm', 'timetamp': start_time})
            except:
                client.retry_connection()
            start_time = time.time()

