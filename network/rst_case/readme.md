# RST case
If we enable SO_LINGER and set linger time to 0 can cause the socket to send a TCP RST (reset) when it's closed.

RST can close the connection directly without four-way close connection. It's used for high performance apps.

RST related:
- if a socket receives RST when there's data remaining in buffer, it becomes CLOSED immediately
- if a socket set SO_LINGER time to 0 and enable it, send RST when close connection. Otherwise, send FIN and server becomes FIN_WAIT1
