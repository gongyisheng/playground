# ss (socket statistics)
### Install
linux: `apt-get install iproute2`  
mac: `brew install iproute2mac`
### Usage
doc: https://man7.org/linux/man-pages/man8/ss.8.html
1. buffer memory usage
memory(buffer): `ss -m`  
tcp: `ss -it`
2. rto
rto: `ss -itn |egrep "cwnd|rto"`
3. filter ip address and port
`ss -ant dst :80`
`ss -ant dst 111.161.68.235`
4. stats
`ss -s`
5. tcp states
`ss -lt`