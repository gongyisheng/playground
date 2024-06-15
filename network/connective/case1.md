# Connective Case 1
## Observation
(remote machine) telnet to a service on a k8s node failed
ping raspberrypi-1.local ---> success
telnet raspberrypi-1.local 9090 ---> success
telnet raspberrypi-1.local 34431 ---> failed
telnet 10.0.0.147 34431 ---> failed
(local machine) telnet to a service on a k8s node success
ping raspberrypi-1.local ---> success
telnet 127.0.0.1 34431 ---> success
telnet raspberrypi-1.local 34431 ---> failed
## reason
bind to 127.0.0.1 instead of 0.0.0.0
```
yisheng@raspberrypi-1:~/.kube$ netstat -tuln
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 127.0.0.1:40329         0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:40729         0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:35425         0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:46193         0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.54:53           0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.53:53           0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:34431         0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:33091         0.0.0.0:*               LISTEN
tcp6       0      0 :::9090                 :::*                    LISTEN
tcp6       0      0 :::9100                 :::*                    LISTEN
tcp6       0      0 :::22                   :::*                    LISTEN
```