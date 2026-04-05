import zmq
import torch

PORT = 3636

ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind(f"tcp://*:{PORT}")


print(f"Python server online, listening on Port: {PORT}")


while True:
    frames = socket.recv_multipart()
    
    batch_size = int.from_bytes(frames[0], "little")
    batch = torch.frombuffer(frames[1], dtype=torch.float32).view(-1, 64, 103)
    
    print(batch_size)
    print(batch.shape)
    
    torch.save(batch, "debug.ts")
    
    socket.send(b"Ok")