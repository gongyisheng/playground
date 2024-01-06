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