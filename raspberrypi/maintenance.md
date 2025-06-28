# Maintenance tips for Raspberry Pi
## Package management
```bash
# install known packages
sudo apt-get update
sudo apt-get upgrade
sudo apt-get autoremove

# install .deb packages
# method 1
sudo dpkg -i package-name.deb
# method 2
sudo apt install ./package-name.deb

# hold a package, prevent upgrade
sudo apt-mark hold code
sudo apt-mark showhold

# clean package cache
sudo apt clean
sudo apt autoclean
```

# setup ssh login
`sudo apt install openssh-server`
`sudo systemctl enable ssh`
`sudo systemctl start ssh`

use public key to login:
```
cd ~/.ssh
vim authorized_keys
<COPY YOUR PUB KEY TO authorized_keys>
```
disable password login:

`sudo vim /etc/ssh/sshd_config`
`PasswordAuthentication no`
`sudo systemctl restart ssh`

## user related
```bash
# create user with home dir:
sudo useradd -m <USER> --groups <GROUP>
# delete password
sudo passwd -d <USER>
# use bash:
bash

# add to sudoers (method1)
visudo /etc/sudoers
# Allow members of group sudo to execute any command
%sudo   ALL=(ALL:ALL) ALL

# add to sudoers(method 2, add to sudo group):
sudo adduser <USER> sudo

# change password
# change password of current user:  
passwd  

# create a new user:  
sudo useradd -m <USERNAME> -G sudo  
sudo passwd <USERNAME>
# delete default pi user:  
sudo deluser pi
# delete user with its home folder  
sudo deluser -remove-home pi
``` 

## Disable sleep mode
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```
ref: https://askubuntu.com/questions/47311/how-do-i-disable-my-system-from-going-to-sleep

## Disable tracker3 for ubuntu desktop
`systemctl --user mask tracker-extract-3.service tracker-miner-fs-3.service tracker-miner-rss-3.service tracker-writeback-3.service tracker-xdg-portal-3.service tracker-miner-fs-control-3.service`  
`tracker3 reset -s -r`  
`vim ~/.config/autostart/tracker-miner-fs-3.desktop`  
```
[Desktop Entry]
Hidden=true
```
ref: https://askubuntu.com/questions/1344050/how-to-disable-tracker-on-ubuntu-20-04

## crontab usage
```bash
# list job
crontab -l
# add cronjob
crontab -e -u <USER>
* * * * * <COMMAND>` # (remember to save log if needed)
# cronjob save log (not recommend)
sudo vim /etc/rsyslog.d/50-default.conf
# cronjob execution record
grep CRON /var/log/syslog
# chmod of file if run with root user
chmod +x <FILE>
```

# log2ram
mount /var/log to memory to avoid disk writes
## install
```
echo "deb [signed-by=/usr/share/keyrings/azlux-archive-keyring.gpg] http://packages.azlux.fr/debian/ bookworm main" | sudo tee /etc/apt/sources.list.d/azlux.list
sudo wget -O /usr/share/keyrings/azlux-archive-keyring.gpg  https://azlux.fr/repo.gpg
sudo apt update
sudo apt install log2ram
```
note: make sure /var/log size is smaller than size in conf. see https://github.com/azlux/log2ram/issues/90  

ref: https://github.com/azlux/log2ram