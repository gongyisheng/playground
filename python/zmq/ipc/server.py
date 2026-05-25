import zmq

def run_server():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("ipc:///tmp/zmq-handlers.sock")
    print("server listening on ipc:///tmp/zmq-handlers.sock")

    while True:
        msg = sock.recv()
        print(f"recv: {msg.decode()}")
        sock.send(f"got: {msg.decode()}".encode())

if __name__ == "__main__":
    run_server()
