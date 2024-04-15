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
[install from source]:  
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

[install from apt-get]:
`sudo apt install lsb-release curl gpg`
`curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg`
`echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list`
`sudo apt-get update`
`sudo apt-get install redis`
ref: https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/