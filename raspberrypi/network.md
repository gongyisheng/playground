# connect to raspberry pi with ethernet
(windows)  
share wlan with ethernet  
find raspberrypi ip:   
`arp -a`  
ref: https://zhuanlan.zhihu.com/p/37761024  

# network speed test
`sudo apt-get install iperf3`   
`iperf3 -s` server side   
`iperf3 -c <server_ip>` client side   
You will see result like this:   
```
yisheng@raspberrypi-5:~$ iperf3 -s -p 8080
------------------------------------------------------------
Server listening on TCP port 8080
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  1] local 10.0.0.142 port 8080 connected with 10.0.0.165 port 59707 (icwnd/mss/irtt=14/1448/9853)
[ ID] Interval       Transfer     Bandwidth
[  1] 0.0000-12.5710 sec  3.97 MBytes  2.65 Mbits/sec
```
client side configs
`iperf3 -c <server_ip> -P 3 -t 30` set concurrency=3, time=30

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
install avahi
`sudo apt update`
`sudo apt install avahi-daemon avahi-utils`
check avahi service status
`sudo systemctl status avahi-daemon`
`sudo systemctl start avahi-daemon`
`sudo systemctl enable avahi-daemon`
check hostname:
`hostname`
update hostname:
`hostnamectl set-hostname server1.example.com`
setup mDNS:
- set `MulticastDNS=yes` in `/etc/systemd/resolved.conf`
- enable mDNS for specific interface: `resolvectl mdns <interface> yes`
- restart resolved: `sudo systemctl restart systemd-resolved`

# wlan quality
`iwconfig wlan0 | grep Quality`   
`cat /proc/net/wireless`   
IMPORTANT: don't put HDD or metal thing too close to the board otherwise it affects wifi quality 

# change wifi
`sudo vim /etc/netplan/50-cloud-init.yaml`

```
# this file generated from information provided by the datasource. Changes
# to it will not persist across an instance reboot. To disable cloud-init’s
# network configuration capabilities, write a file 
# /etc/cloud/cloud.cfg.d/99-disable-network-config-cfg with the following:
# network:{config: disabled}
network:
    ethernets:
        eth0:
            dhcp4: true
            optional: true
    version: 2
    wifis:
        wlan0:
            access-points:
                WIFI-NAME:
                    password: WIFI-PASSWORD
            dhcp4: true
            optional: true
```
ref: https://raspberrypi.stackexchange.com/questions/111722/rpi-4-running-ubuntu-server-20-04-cant-connect-to-wifi
Note: If want to connect to ethernet, just remove wifis