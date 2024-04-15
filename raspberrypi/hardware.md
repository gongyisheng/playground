# cpu info
`lscpu`  
`cat /proc/cpuinfo`  

# memory info
`free -m`  
`free -h`  

# setup swap
flush existing swap  
`sudo swapoff -a`  
add new swap files (bs=size per file, count=file count, bs*count=swap size)  
`sudo dd if=/dev/zero of=/var/swap bs=4M count=1024`  
initalize(format) swap files  
`sudo mkswap /var/swap`  
turn on swap  
`sudo swapon /var/swap`  
edit `/etc/fstab` file, start swap after boot  
`/var/swap swap swap defaults 0 0`  

# disk info
`df -h`

# stress test
`sudo apt-get install stress`
`while true; do vcgencmd measure_clock arm; vcgencmd measure_temp; sleep 10; done& stress -c 4 -t 900s`