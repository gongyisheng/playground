## start redis
### redis 1
```
docker run -d \
--name redis-1 \
--net="host" \
redis:7.2 --port 6379
```
### redis 2
```
docker run -d \
--name redis-2 \
--net="host" \
redis:7.2 --port 6380
```

## set up lvs
`sudo apt-get install ipvsadm`
`sudo ipvsadm -A -t 127.0.0.1:8080 -s rr`
`sudo ipvsadm -a -t 127.0.0.1:8080 -r 127.0.0.1:6379 -m`
`sudo ipvsadm -a -t 127.0.0.1:8080 -r 127.0.0.1:6380 -m`

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
`tc qdisc add dev lo root netem delay 200ms`