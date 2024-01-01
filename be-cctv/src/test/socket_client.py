import socketio
import time

sio = socketio.Client()


@sio.on('alert')
def on_message(data):
    print('HandShake', data)

sio.connect('ws://localhost:8080', socketio_path="/ws/socket.io", transports=["websocket", "polling"])
while True:
    time.sleep(2)
    # ... connect to a server and use the client
    # ... no need to manually disconnect!