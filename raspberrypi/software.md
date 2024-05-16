# update and upgrade
`sudo apt-get update`  
`sudo apt-get upgrade`  

# install vim, tcpdump, perf, network tools
`sudo apt-get install vim tcpdump linux-tools-common linux-tools-generic dnsutils net-tools wireless-tools build-essential`  

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

# setup docker
ref: https://docs.docker.com/engine/install/ubuntu/

# setup grafana
ref: https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/

# setup node-exporter, prometheus
ref: https://easycode.page/monitoring-on-raspberry-pi-with-node-exporter-prometheus-and-grafana/#steps-to-deploy-prometheus-on-docker

# run monitoring
- run node exporter (docker native)
```
# Navigate to the node-exporter directory
cd monitoring/node-exporter
# Run the node-exporter docker container
docker run -d \
--name="node-exporter" \
--net="host" \
--pid="host" \
-v "/:/host:ro,rslave" \
--restart=always \
quay.io/prometheus/node-exporter:latest --path.rootfs=/host
# Node exporter is installed on 9100 port by default
```

- run prometheus
1. configure yaml file:
```
# Edit the file using a file editor (nano is this case)
vim prometheus.yml
# Add the below content to the file
global:
  scrape_interval: 5s
  external_labels:
    monitor: 'node'
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090'] ## IP Address of the localhost
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100'] ## IP Address of the localhost
```
2. start:
```
docker run -d \
--name prometheus \
--net="host" \ 
--pid="host" \
-p 9090:9090 \
-v ~/monitoring/prometheus:/etc/prometheus \
--restart always \
prom/prometheus
```
**note**: net="host" means container to use host's network configuration. 

# run cloudflare tunnel - ubuntu
```
docker run -d \
--name cloudflared \
--restart=always \
--net="host" \
--pid="host" \
cloudflare/cloudflared:latest tunnel --no-autoupdate run --token
```

# run cloudflare tunnel - arm32
```
download from: https://hobin.ca/cloudflared/
tar -xvzf cloudflared-stable-linux-arm.tgz
sudo cp ./cloudflared /usr/local/bin
sudo chmod +x /usr/local/bin/cloudflared
```

# run sqlite web
```
sudo nohup sqlite_web -p 6666 <YOUR DB FILE> -x -P -l <YOUR LOG FILE>
```

# run sqlite web in docker
```
docker run -d \
--name sqlite-web \
-p 6666:8080 \
-v /path/to/your-data:/data \
-e SQLITE_DATABASE=db_filename.db \
-e SQLITE_WEB_PASSWORD=<YOUR PASSWORD> \
coleifer/sqlite-web
```

# run dufs
```
docker run -d \
--name dufs \
--restart always \
-v /media/usbdisk/:/data \
-v /media/usbdisk/codebase/user-key/dufs/dufs.yaml:/etc/dufs/dufs.yaml \
-p 5000:5000 \
sigoden/dufs --config /etc/dufs/dufs.yaml
```
