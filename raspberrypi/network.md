# connect to raspberry pi with ethernet
(windows)  
share wlan with ethernet  
find raspberrypi ip:   
`arp -a`  
ref: https://zhuanlan.zhihu.com/p/37761024  

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

# setup DuckDNS for dynamic DNS
start cron:  
`sudo service cron start`  
ref: https://www.duckdns.org/install.jsp  

# setup caddy for HTTPS
ref: https://github.com/caddyserver/caddy