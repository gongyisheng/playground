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
`sudo -u postgres createuser -s <username> --password`  
`sudo -u postgres createdb <dbname>`  

## connect to database
### as normal user
`psql --user <username> --dbname <dbname> --host localhost`
### as admin user
`sudo -u postgres psql template1`
### drop user
`DROP USER <username>;`
### connect using cloudflare tunnel (only for test usage)
`cloudflared access tcp --hostname <your host> --url localhost:5432`
`psql --user yisheng --host localhost --password --dbname test`

### uninstall postgresql
`sudo apt-get --purge remove postgresql postgresql-*`
`dpkg -l | grep postgres` ---> for double check
`sudo apt-get --purge remove postgresql postgresql-doc postgresql-common` ---> remove packages

## ref
- https://ubuntu.com/server/docs/install-and-configure-postgresql