# update and upgrade
`sudo apt update`  
`sudo apt upgrade`  

# install vim, tcpdump, perf, network tools
`sudo apt install vim tcpdump linux-tools-common linux-tools-generic dnsutils net-tools wireless-tools build-essential python3-pip redis-tools`  

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

# restart all running docker containers
`sudo docker restart $(sudo docker ps -q)`

# clean docker unused image
`sudo docker image ls`  
`sudo docker image prune -a`  

# clean docker system
`sudo docker system prune -a`

# enter docker container env
`docker exec -it <container_name> sh`

# setup grafana
ref: https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/

# setup node-exporter, prometheus
ref: https://easycode.page/monitoring-on-raspberry-pi-with-node-exporter-prometheus-and-grafana/#steps-to-deploy-prometheus-on-docker

# run docker compose
```
docker compose -f docker-compose.yaml up -d
```

# find .DS_Store and delete
```
find . -name ".DS_Store" -delete
```

# run ffmpeg
```
sudo apt install ffmpeg
// extreme compress
ffmpeg -i "input.mp4" -c:v libx264 -tag:v avc1 -movflags faststart -crf 30 -preset superfast "output.mp4"
// add subtitle (hard subtitle)
ffmpeg -i input.mkv -vf subtitles=subtitles.srt output.mkv
// add subtitle (soft subtitle)
ffmpeg -i input.mkv -i subtitles.srt -c copy output.mkv
// cut video from [start] to [end] (eg, 00:00:00 to 1:00:00)
ffmpeg -ss [start] -to [end] -accurate_seek -i [in].mp4 -codec copy -avoid_negative_ts 1 -map_metadata -1 [out].mp4
// delay subtitle by seconds
ffmpeg -itsoffset [num] -i subtitles.srt -c copy subtitles_delayed.srt
// solve 6 channel issue (Unsupported channel layout "6 channels")
ffmpeg -y -i source.avi -acodec copy source.6.aac
faad -d -o source.2.pcm source.6.aac
ffmpeg -y -i source.avi -i source.2.pcm -map 0:0 -map 1:0 -vcodec copy -acodec copy output.avi
ref: https://lzone.de/blog/ffmpeg-AAC-Can-not-resample-6-channels
```

# run bypy (baiduyun python client)
download from background:  
```
screen -S bypy download <cloud path> <local path>
ctrl + a + d
```

# install pinyin on raspberrypi OS
```
# 1. add chinese locale (zh_CN.UTF-8)
sudo dpkg-reconfigure locales

# 2. install facitx
sudo apt install fcitx fcitx-libpinyin

# 3. config os
Raspberry Pi Menu > Preferences > Input Method

- First screen:
Current confituration for the input method...
click [OK]

- Second screen:
Do you explicetly select the user configuration?
click [YES]

- Third screen:
Select: (*) fcitx activate Flexible Input Method Framework (fcitx) @
click [OK]

# 4. reboot

# 5. facitx config
- add "Pinyin (LibPinyin)"
- select 'Pinyin (LibPinyin)'
```
ref: https://forums.raspberrypi.com/viewtopic.php?t=222801

# install wechat linux
```
# download deb, install
https://linux.weixin.qq.com/en

# resolve libtiff.so.5 issue on ubuntu 22/24
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libtiff.so.6 libtiff.so.5
```


