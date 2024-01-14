# update and upgrade
`sudo apt-get update`  
`sudo apt-get upgrade`  

# install vim, tcpdump, perf
`sudo apt-get install vim tcpdump linux-tools-common linux-tools-generic`  

# setup mysql
download and install:  
`sudo apt install mariadb-server`  
`sudo mysql_secure_installation`  
start:  
`sudo service mysql start`  
change password:  
`SET PASSWORD FOR 'root'@'localhost' = PASSWORD('<newpass>');`  
ref: https://pimylifeup.com/raspberry-pi-mysql/

# setup redis
download:  
`wget https://download.redis.io/redis-stable.tar.gz`
compile:  
`tar -xzvf redis-stable.tar.gz`  
`cd redis-stable`  
`make`  
install:  
`sudo make install`  
start:  
`redis-server`  
ref:https://redis.io/docs/install/install-redis/install-redis-from-source/
