# RST case
If we enable SO_LINGER and set linger time to 0 can cause the socket to send a TCP RST (reset) when it's closed.

RST can close the connection directly without four-way close connection. It's used for high performance apps.

RST related:
- if a socket receives RST when there's data remaining in buffer, it becomes CLOSED immediately
- if a socket set SO_LINGER time to 0 and enable it, send RST when close connection. Otherwise, send FIN and server becomes FIN_WAIT1

When running client.py, server side can see following log
```
Traceback (most recent call last):
  File "/usr/lib/python3.11/socketserver.py", line 691, in process_request_thread
    self.finish_request(request, client_address)
  File "/usr/lib/python3.11/http/server.py", line 1306, in finish_request
    self.RequestHandlerClass(request, client_address, self,
  File "/usr/lib/python3.11/http/server.py", line 667, in __init__
    super().__init__(*args, **kwargs)
  File "/usr/lib/python3.11/socketserver.py", line 755, in __init__
    self.handle()
  File "/usr/lib/python3.11/http/server.py", line 432, in handle
    self.handle_one_request()
  File "/usr/lib/python3.11/http/server.py", line 400, in handle_one_request
    self.raw_requestline = self.rfile.readline(65537)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/socket.py", line 706, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
ConnectionResetError: [Errno 104] Connection reset by peer
```
