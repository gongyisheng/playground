# connect to raspberry pi with ethernet
(windows)  
share wlan with ethernet  
find raspberrypi ip:   
`arp -a`  
ref: https://zhuanlan.zhihu.com/p/37761024  

# connect to raspberry pi with ssh
connect with ssh:  
`ssh pi@<ip>`  

# setup DuckDNS for dynamic DNS
start cron:  
`sudo service cron start`  
ref: https://www.duckdns.org/install.jsp  

# setup caddy for HTTPS
ref: https://github.com/caddyserver/caddy