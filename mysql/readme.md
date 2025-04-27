# MySQL install
```
# on ubuntu
sudo apt install mysql-server mysql-client
# on raspberrypi os
sudo apt install mariadb-server mariadb-client
```

# MySQL configuration
## connect to mysql
`mysql -h 127.0.0.1 -u root`
## reset root user passowrd
`ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Password you want to use';`
## add dev user
```
CREATE USER 'dev'@'%' IDENTIFIED BY 'dev';
GRANT ALL PRIVILEGES ON *.* TO 'dev'@'%';
```
## configuration
```
vim /etc/mysql/my.cnf

# add following lines

[mysqld]
bind-address = 0.0.0.0  # allow remote access
max_connections = 1025  # default is 151
```