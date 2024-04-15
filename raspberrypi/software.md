# update and upgrade
`sudo apt-get update`  
`sudo apt-get upgrade`  

# install vim, tcpdump, perf
`sudo apt-get install vim tcpdump linux-tools-common linux-tools-generic dnsutils`  

# setup mariadb
download and install:  
`sudo apt install mariadb-server`
`sudo mysql_secure_installation`  
start:  
`sudo service mysql start`  
change password:  
`SET PASSWORD FOR 'root'@'localhost' = PASSWORD('<newpass>');`  
ref: https://pimylifeup.com/raspberry-pi-mysql/

# setup mysql
download and install:
`sudo apt install mysql-server`
check status
`sudo service mysql status`
create user:
`CREATE USER 'root'@'localhost' IDENTIFIED BY 'password';`
grant permission:
`GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' WITH GRANT OPTION;`
`GRANT CREATE, ALTER, DROP, INSERT, UPDATE, DELETE, SELECT, REFERENCES, RELOAD on *.* TO 'root'@'localhost' WITH GRANT OPTION;`
show permission:
`SHOW GRANTS FOR 'username'@'host';`
ref: https://ubuntu.com/server/docs/databases-mysql
ref: https://www.digitalocean.com/community/tutorials/how-to-create-a-new-user-and-grant-permissions-in-mysql

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
ref: https://redis.io/docs/install/install-redis/install-redis-from-source/
