# cpu info
`lscpu`  
`cat /proc/cpuinfo`  

# memory info
`free -m`  
`free -h`  

# setup swap
flush existing swap  
`swapoff -a`  
add new swap files (bs=size per file, count=file count, bs*count=swap size)  
`dd if=/dev/zero of=/var/swap bs=4M count=1024`  
initalize(format) swap files  
`mkswap /var/swap`  
turn on swap  
`swapon /var/swap`  
edit /etc/fstab  
`/var/swap swap swap defaults 0 0`  

# disk info
`df -h`