# connect to raspberry pi with ethernet
(windows)  
share wlan with ethernet  
find raspberrypi ip:   
`arp -a`  
ref: https://zhuanlan.zhihu.com/p/37761024  

# network speed test
`sudo apt-get install iperf`   
`iperf -s -p 8080` server side   
`iperf -c <server_ip> -p 8080` client side   
You will see result like this:   
```
yisheng@raspberrypi-5:~$ iperf -s -p 8080
------------------------------------------------------------
Server listening on TCP port 8080
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  1] local 10.0.0.142 port 8080 connected with 10.0.0.165 port 59707 (icwnd/mss/irtt=14/1448/9853)
[ ID] Interval       Transfer     Bandwidth
[  1] 0.0000-12.5710 sec  3.97 MBytes  2.65 Mbits/sec
```

# static ip address
update static ip address, router, dns:  
`sudo vim /etc/dhcpcd.conf`  
restart dhcpcd:  
`sudo service dhcpcd restart`  
ping test on google:  
`ping google.com`  
dns resolve test on google:  
`dig google.com`  
ref: https://droidyue.com/blog/2020/05/01/set-dns-server-on-reaspberry-pi/

# connect to raspberry pi with ssh
connect with ssh:  
`ssh pi@<ip>`  

# firewall
install ufw:  
`sudo apt-get install ufw`  
set inbound rules:  
`sudo ufw allow 22/tcp`  
`sudo ufw allow 80/tcp`  
`sudo ufw allow 443/tcp`  
`sudo ufw allow 8000/tcp`  
`sudo ufw allow 10000:11000/tcp` (allow by port range)  
`sudo ufw allow from 192.168.0.1` (allow by ip)  
delete rules:  
`sudo ufw delete allow 8000`  
`sudo ufw delete allow 10000:11000/tcp`  
`sudo ufw delete allow from 192.168.0.1`
enable firewall  
`sudo ufw enable`  
disable firewall  
`sudo ufw disable`  
show status  
`sudo ufw status`  
ref: https://121rh.com/pc/raspberry/ufw/

# setup DuckDNS for dynamic DNS
start cron:  
`sudo service cron start`  
ref: https://www.duckdns.org/install.jsp  

# setup caddy for HTTPS
ref: https://github.com/caddyserver/caddy

# setup mDNS
check hostname:
`hostname`
update hostname:
`hostnamectl set-hostname server1.example.com`
setup mDNS:
- set `MulticastDNS=yes` in `/etc/systemd/resolved.conf`
- enable mDNS for specific interface: `resolvectl mdns <interface> yes`
- restart resolved: `sudo systemctl restart systemd-resolved`