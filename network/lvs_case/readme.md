# LVS case
## run redis in docker
`sudo docker compose -f redis-docker-compose.yaml up -d`

## set up lvs
`sudo apt-get install ipvsadm`  
`sudo ipvsadm -A -t 1.2.3.4:6379 -s rr`  
`sudo ipvsadm -a -t 1.2.3.4:6379 -r 10.0.0.197:6379 -m` ---> add real redis server 1  
`sudo ipvsadm -a -t 1.2.3.4:6379 -r 10.0.0.142:6379 -m` ---> add real redis server 2  
`sudo bash -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'` ---> enable ip forwarding  
ref: https://linux.die.net/man/8/ipvsadm   

## check lvs status
`sudo ipvsadm -L -n`
```
yisheng@raspberrypi-1:~$ sudo ipvsadm -L -n
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
TCP  127.0.0.1:8080 rr
  -> 127.0.0.1:6379               Masq    1      0          0
  -> 127.0.0.1:6380               Masq    1      0          0
```

## delay network
`ifconfig` ---> get the network interface name you want to delay  
`sudo tc qdisc add dev wlp4s0 root netem delay 100ms` ---> delay 100ms on wlp4s0  
Here I use 10.0.0.197 as the instance to delay.

## set redis timeout
`redis-cli config set timeout 5` ---> set redis timeout to 5s (idle connection will be closed after 5s)

## run python script
`pip install redis`  
`python3 polling.py`  
The script uses a connection pool with size=200.
It run polling on the redis server every 3s, with concurrency either = 195 or 200

## clean up
`sudo ipvsadm -D -t 127.0.0.1:8080`  
`sudo tc qdisc del dev wlp4s0 root`  