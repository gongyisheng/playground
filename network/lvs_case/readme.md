## start redis
### run redis in docker
```
docker run -d \
--name redis \
--net="host" \
redis:7.2 --port 6379
```

## set up lvs
`sudo apt-get install ipvsadm`
`sudo ipvsadm -A -t 1.2.3.4:6379 -s rr`
`sudo ipvsadm -a -t 1.2.3.4:6379 -r 10.0.0.197:6379 -m`
`sudo ipvsadm -a -t 1.2.3.4:6379 -r 10.0.0.142:6379 -m`
`sudo bash -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'`

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

## run script
`sudo tc qdisc add dev wlp4s0 root netem delay 2000ms`

## clean up
`sudo ipvsadm -D -t 127.0.0.1:8080`
`sudo tc qdisc del dev wlp4s0 root`