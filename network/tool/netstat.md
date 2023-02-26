### basic
`netstat -an`: display all active network connections
`netstat -an p`: display all active network connections with PID/Program name
`netstat -r`: display the routing table
`netstat -s`: display network statistics
`netstat -i`: display per-interface statistics
`netstat -m`: display kernel resident modules

### filter result based on connection state
`netstat -an | grep LISTEN`: display all listening ports
`netstat -an | grep ESTABLISHED`: display all established connections
`netstat -an | grep TIME_WAIT`: display all connections in TIME_WAIT state
`netstat -an | grep CLOSE_WAIT`: display all connections in CLOSE_WAIT state
`netstat -an | grep SYN_SENT`: display all connections in SYN_SENT state
`netstat -an | grep SYN_RECV`: display all connections in SYN_RECV state
`netstat -an | grep FIN_WAIT1`: display all connections in FIN_WAIT1 state
`netstat -an | grep FIN_WAIT2`: display all connections in FIN_WAIT2 state
`netstat -an | grep LAST_ACK`: display all connections in LAST_ACK state
`netstat -an | grep CLOSING`: display all connections in CLOSING state
`netstat -an | grep UNKNOWN`: display all connections in UNKNOWN state

### filter result based on protocol
`netstat -an | grep tcp`: display all TCP connections
`netstat -an | grep udp`: display all UDP connections
`netstat -an | grep kevt`: display all kernel event sockets
`netstat -an | grep kctl`: display all kernel control sockets
