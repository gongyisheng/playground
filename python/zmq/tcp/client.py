import zmq

def setup():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://localhost:5555")
    return sock

def run_send():
    sock = setup()
    msg = "hello, world".encode()
    sock.send(msg)
    reply = sock.recv()
    print(reply.decode())


if __name__ == "__main__":
    run_send()