import zmq

def run_server():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://localhost:5555")
    print("server listening on tcp://localhost:5555")

    while True:
        msg = sock.recv()
        print(f"recv: {msg.decode()}")
        sock.send(f"got: {msg.decode()}".encode())

if __name__ == "__main__":
    run_server()
