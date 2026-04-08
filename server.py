import zmq
import torch
import warnings
from Model import ChessModel

PORT = 3636

ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind(f"tcp://*:{PORT}")

socket.setsockopt(zmq.RCVTIMEO, 1000)

# Create the model/ set to eval mode
model = ChessModel()
model.eval()


print(f"Python server online, listening on Port: {PORT}")

try:
    while True:
        try:
            frames = socket.recv_multipart()
        except zmq.Again:
            continue
        
        batch_size = int.from_bytes(frames[0], "little")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = torch.frombuffer(frames[1], dtype=torch.float32).view(-1, 64, 103).to(device='cuda') # Move to GPU
        
        print(batch_size)
        print(batch.shape)
        
        # torch.save(batch, "debug.ts")
        with torch.no_grad():
            policy, value = model(batch)
        
        print(policy.shape)
        print(value.shape)
        
        socket.send(b"Ok")
except KeyboardInterrupt:
    print("\nShutting down server...")
finally:
    socket.close()