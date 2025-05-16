import socket
import struct
import time

# Define constants for SO_LINGER
LINGER_TIME = 0  # Set linger time to 0 seconds for an immediate reset

def create_socket():
    # Create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Set SO_LINGER option to send a RST upon closing
    linger_option = struct.pack('ii', 1, LINGER_TIME)  # 1 enables linger, 0 is the linger time
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger_option)
    
    return sock

def client():
    # Create socket and connect to server
    sock = create_socket()
    server_address = ('localhost', 8080)
    sock.connect(server_address)
    
    try:
        # Send some data to the server
        message = b'This is a test message.'
        sock.sendall(message)
        
        # Sleep for a while (to simulate some interaction)
        time.sleep(2)
        
    finally:
        # Close the socket (RST should be sent automatically)
        print("Closing socket and sending RST...")
        sock.close()

if __name__ == "__main__":
    client()

