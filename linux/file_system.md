linux file system: 
- /bin: contains the most important binaries for the system
- /sbin: contains the most important binaries for the system
- /dev: contains the device files
- /etc: contains the configuration files
- /home: contains the home directories of the users
- /lib: contains the libraries
- /media: contains the mount points for removable media
- /mnt: contains the mount points for removable media
- /opt: contains the optional software
- /proc: contains the process information
- /root: contains the home directory of the root user
- /run: contains the runtime data
- /srv: contains the data for services
- /sys: contains the system information
- /tmp: contains the temporary files
- /usr: contains the user binaries, libraries and other data
- /var: contains the variable data

environment variables:
- $HOME: the home directory of the current user
- $PATH: the list of directories where the shell looks for commands  
ls, pwd, cat, these are all executable files, and the shell looks for them in the directories listed in the PATH variable

comman configuration files:
- /etc/passwd: contains the user accounts
- /etc/shadow: contains the encrypted passwords
- /etc/group: contains the groups
- /etc/hosts: contains the host names and their IP addresses
- /etc/hostname: contains the host name
- /etc/resolv.conf: contains the DNS servers
- /etc/fstab: contains the mount points for the file systems
- /etc/issue: contains the message that is displayed when the system boots
- /etc/motd: contains the message that is displayed when the user logs in
- /etc/profile: contains the configuration for the shell
- /etc/bash.bashrc: contains the configuration for the bash shell
- /etc/sudoers: contains the configuration for the sudo command

common device files:
- /dev/null: Bottomless garbage can
- /dev/fd: a special file that represents the first floppy disk drive
- /dev/mem: a special file that represents the physical memory
- /dev/hd: a special file that represents the first hard disk
- /dev/mouse: a special file that represents the mouse
- /dev/usb: a special file that represents the USB devices
- /dev/par: a special file that represents the parallel port
- /dev/radio: a special file that represents the radio
- /dev/vedio: a special file that represents the video

files in /tmp are managed by OS, can be deleted after reboot  
files in /var/tmp are managed by user programs, cannot be deleted after reboot

ls -l: list the files in the current directory, with the details  
ls -a: list the files in the current directory, including the hidden files  
ls -t: list the files in the current directory, sorted by the time of the last modification
ls -r: list the files in the current directory, in reverse order

find `<path>` -name `<pattern>`: find the files in the specified path that match the specified pattern  
find `<path>` -type `<type>`: find the files in the specified path that match the specified type  
find `<path>` -size `<size>`: find the files in the specified path that match the specified size  
find -exec `<command>` {}: execute the specified command for each file found
