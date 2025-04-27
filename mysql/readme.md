# connect to mysql
`mysql -h 127.0.0.1 -u root`
# reset root user passowrd
`ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Password you want to use';`
# add dev user
```
CREATE USER 'dev'@'%' IDENTIFIED BY 'dev';
GRANT ALL PRIVILEGES ON *.* TO 'dev'@'%';
```