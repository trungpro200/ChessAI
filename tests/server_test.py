import zmq
import time

def stress_test_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://localhost:3636")

    print("Python server ready. Waiting for junk...")
    
    count = 0
    while True:
        # Receive multipart
        msg = socket.recv_multipart() # msg[0] is header, msg[1] is data
        
        # Simulate a tiny bit of "processing" delay (0.5ms)
        # time.sleep(0.0005) 
        
        # Send back a small response (simulated Value/Policy)
        socket.send(b"OK")
        
        count += 1
        if count % 10 == 0:
             print(f"Processed {count} batches")

if __name__ == "__main__":
    stress_test_server()