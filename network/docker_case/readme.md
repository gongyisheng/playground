# docker case
## question 1
Make sure 2 containers can connect to each other

## answer 1
use docker command line
```
sudo docker network create test_network

sudo docker run -d --name c1 --network test_network nginx
sudo docker run -d --name c2 --network test_network nginx

docker exec -it c1 sh
apt update -y
apt install iputils-ping -y
ping c2
```

result  
```
# ping c2
PING c2 (172.20.0.3) 56(84) bytes of data.
64 bytes from c2.test_network (172.20.0.3): icmp_seq=1 ttl=64 time=0.096 ms
64 bytes from c2.test_network (172.20.0.3): icmp_seq=2 ttl=64 time=0.063 ms
64 bytes from c2.test_network (172.20.0.3): icmp_seq=3 ttl=64 time=0.057 ms
64 bytes from c2.test_network (172.20.0.3): icmp_seq=4 ttl=64 time=0.052 ms
64 bytes from c2.test_network (172.20.0.3): icmp_seq=5 ttl=64 time=0.049 ms
64 bytes from c2.test_network (172.20.0.3): icmp_seq=6 ttl=64 time=0.065 ms
^C
--- c2 ping statistics ---
6 packets transmitted, 6 received, 0% packet loss, time 5116ms
rtt min/avg/max/mdev = 0.049/0.063/0.096/0.015 ms
```

You can also double check by running `ip addr`
```
apt install iproute2 -y
ip addr
```

result
```
# ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: eth0@if9: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default 
    link/ether 16:02:14:3b:d1:01 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.20.0.2/16 brd 172.20.255.255 scope global eth0
       valid_lft forever preferred_lft forever
```

## question 2
If a container is attached to multiple docker network, what will it be like

## answer 2
```
sudo docker network create test_network1
sudo docker network create test_network2

sudo docker run -d --name c1 --network test_network1 nginx
sudo docker run -d --name c2 --network test_network1 --network test_network2 nginx
sudo docker run -d --name c3 --network test_network2 nginx

docker exec -it c1 sh
```

result
```
# c2 can ping both c1 and c3
# c1 can ping c2, but can't ping c3
# c3 can ping c2, but can't ping c1

# for c2, you can see 2 network interfaces
# ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: eth0@if24: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default 
    link/ether 4e:1a:02:1f:51:7a brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.21.0.2/16 brd 172.21.255.255 scope global eth0
       valid_lft forever preferred_lft forever
3: eth1@if25: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default 
    link/ether 6e:23:10:04:02:a3 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.20.0.3/16 brd 172.20.255.255 scope global eth1
       valid_lft forever preferred_lft forever
```

## docker network
```
bridge	The default network driver.
host	Remove network isolation between the container and the Docker host.
none	Completely isolate a container from the host and other containers.
overlay	Overlay networks connect multiple Docker daemons together.
ipvlan	IPvlan networks provide full control over both IPv4 and IPv6 addressing.
macvlan	Assign a MAC address to a container.
```
ref: https://docs.docker.com/engine/network/