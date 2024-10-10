# Nagle algorithm related
By default, nagle algorithm is not used by kernel now. `TCP_NODELAY = 1`  
If we enable nagle algorithm, small messages will wait in the buffer until the buffer size reaches MSS or receive ACK. There'll be message delays.  
The algorithm is against tcp protocol design (based on message stream and it's upper level protocol to decide package length)  