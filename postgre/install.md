# install on ubuntu

## install postgresql server
`sudo apt-get install postgresql`

## install postgresql client
`sudo apt-get install postgresql-client`

## change configuration
`sudo vim /etc/postgresql/*/main/postgresql.conf`, *=version  
`listen_addresses = '*'`  
port can be 5432, which is the default port

## restart server, enable auto restart
`sudo systemctl restart postgresql`  
`sudo systemctl enable postgresql`  

## create user and database
`sudo -u postgres createuser -s <username> -password`  
`sudo -u postgres createdb <dbname>`  

## connect to database
### as normal user
`psql -U <username> -d <dbname> -h localhost`
### as admin user
`sudo -u postgres psql template1`
### drop user
`DROP USER <username>;`