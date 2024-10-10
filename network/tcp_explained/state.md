# State
LISTEN - represents waiting for a connection request from any remote TCP and port.  
SYN-SENT - represents waiting for a matching connection request after having sent a connection request.  
SYN-RECEIVED - represents waiting for a confirming connection request acknowledgment after having both received and sent a connection request.  
ESTABLISHED - represents an open connection, ready to transmit and receive data segments.  
FIN-WAIT-1 - represents waiting for a connection termination request from the remote TCP, or an acknowledgment of the connection termination request previously sent.  
FIN-WAIT-2 - represents waiting for a connection termination request from the remote TCP.  
TIME-WAIT - represents waiting for enough time to pass to be sure the remote TCP received the acknowledgment of its connection termination request.  
CLOSE-WAIT - represents waiting for a connection termination request from the local user.  
CLOSING - represents waiting for a connection termination request acknowledgment from the remote TCP.  
CLOSED - represents no connection state at all.  

![image](tcp_state_diagram.jpg)

## Steps to establish
SYN, ACK, SYN  

```
Q1: Why need 3 way handshake?
A1: If only 1 way handshake, the client do not know whether its message to server is received.
    If only 2 way handshake, the server do not know whether its message to server is received.
    3 way handshake is the minimum requirement for both client and server know that the connection can be established.
    But in theory there're problems like, the client does not know the third SYN to server can be received, it's an engineering design. 
```

## Steps to close
FIN, ACK, FIN, ACK  

State
proactively closed: FIN_WAIT_1, FIN_WAIT_2, TIME_WAIT  
passively closed: CLOSE_WAIT, LAST_ACK  

Note: the proactively closed client need to wait 2 MSL before close the connection. It's 2min by default on linux.  

![image](tcp_close.png)

```
Q1: Why TIME_WAIT takes 2 MSL to close?
A1: To make sure that the connection can be gracefully closed. 
    1. There could still be delayed packets in the network that may arrive after the connection has been terminated after a connection is closed. If the connection is opened again, stale packets may cause confusions.
    2. Make sure the last ack sent to server can be received. Otherwise it may cause new connection fail to establish.
    Why 2 MSL? Because in the worst cause both the packet sent from client is delayed and the ack packet sent from server is delayed. So it's 2 MSL.
    This limits the number that we can establish connections with one target ip (server). We can only establish about 28,232 connections in one minute to target ip. 

Q2: Ways to improve TIME_WAIT
A2: 1. Set SO_LINGER to 0, and the connection will send RST to close the connection and abort all the data in the buffer instead.
    2. Set net.ipv4.tcp_tw_reuse to 1, allow the kernel to reuse connection in TIME_WAIT status based on timestamp
    3. Change net.ipv4.ip_local_port_range, allow kernel to use more ports.

Q3: Why 4 way close?
A3: 2 way for client to close connection. 2 way for server to close connection.

Q4: Explain net.ipv4.tcp_tw_reuse
A4: Allow kernal to reuse connections in TIME_WAIT status. It can be used only when both server and client enables timestamps. It only works for client (only client has TIME_WAIT status). It recycles connection in 1s, allowing the limitation of opening/closing short connections to 30k-60k per second.

Q5: Explain timestamps in TCP
A5: Timestamps are like sequence number and it's strict incremental. It allows server/client to deal with packets out of bind. New connection packets has bigger timestamps. It allows TIME_WAIT connection reuse without waiting 2 MSL.

Q6: Explain tcp_tw_recycle and why it's removed from linux.
A6: One significant problem emerged in environments utilizing Network Address Translation (NAT) or Load Balancing (LB). The feature's aggressive socket reusing led to non-monotonic TCP timestamps, causing disruptions in connections within such environments. As a result, the intended benefits of tcp_tw_recycle were overshadowed by its potential to break connections and create a less stable networking environment.
```