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

# disk health check
`sudo badblocks -s -v /dev/sda`

# stress test
`sudo apt-get install stress`
`while true; do vcgencmd measure_clock arm; vcgencmd measure_temp; sleep 10; done& stress -c 4 -t 900s`

# external disk mount
`sudo fdisk -l` see drivers number  
`sudo blkid` get UUID of a device
`sudo mount /dev/sda# /media/usbdisk` mount disk  
add following line to `/etc/fstab` to mount disk after boot  
`UUID=XXXXXXX /media/usbdisk auto nosuid,nodev,nofail 0 2`
unmount: IMPORTANT! dont use lazy umount if you cares driver can safely unplugged

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