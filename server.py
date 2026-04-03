import zmq
import numpy as np

PORT = 3636

ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind(f"tcp://*:{PORT}")


print(f"Python server online, listening on Port: {PORT}")


while True:
    frames = socket.recv_multipart()
    
    print(frames[0])
    print(frames[1])
    
    socket.send(b"Ok")