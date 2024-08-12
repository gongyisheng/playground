# cpu info
`lscpu`  
`cat /proc/cpuinfo`  

# memory info
`free -m`  
`free -h`  

# setup swap
change default system-level swap size (only for raspberry pi os)  
`sudo vim /etc/dphys-swapfile`  
`CONF_SWAPFILE=4096`(MB)  
`CONF_MAXSWAP=8192`(MB)   
flush existing swap  
`sudo swapoff -a`  
add new swap files (bs=size per file, count=file count, bs*count=swap size)  
`sudo dd if=/dev/zero of=/var/swap bs=4M count=1024`  
initalize(format) swap files  
`sudo mkswap /var/swap`  
get partition uuid   
`sudo blkid /var/swap`  
turn on swap  
`sudo swapon /var/swap` or `sudo swapon -U <UUID>`  
edit `/etc/fstab` file, start swap after boot  
`UUID=<UUID> none swap sw 0 0`  
reboot
`sudo reboot`

# disk info
`df -h`

# stress test
`sudo apt-get install stress`
`while true; do vcgencmd measure_clock arm; vcgencmd measure_temp; sleep 10; done& stress -c 4 -t 900s`

# external disk mount
`sudo fdisk -l` see drivers number  
`sudo blkid` get UUID of a device
`sudo mount /dev/sda# /media/usbdisk` mount disk  
add following line to `/etc/fstab` to mount disk after boot  
`UUID=XXXXXXX /media/usbdisk auto nosuid,nodev,nofail 0 0`
unmount: IMPORTANT! dont use lazy umount if you cares driver can safely unplugged

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
        etho:
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