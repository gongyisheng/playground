# connect to mysql
`mysql -h 127.0.0.1 -u root -p`
# reset root user passowrd
`sudo mysql`
`ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Password you want to use';`