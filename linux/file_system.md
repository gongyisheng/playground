# linux file system:

## permission
`drwxr-xr-x` 
1. d(1): file type, d=directory, -=file
2. rwx(2-4): owner's permission
3. r-x(5-7): user at same group's permission
4. r-x(8-0): other's permission

## change permission
`chgrp`: change group (name should be in `/etc/group`)   
`chgrp -R <GROUP> <DIR>`   
`chown`: change owner (name should be in `/etc/passwd`)    
`chown -R <USER>:<GROUP> <DIR>`  
`chmod`: change permission (r=4, w=2, x=1, sum it up)  
`chmod -R 700 <DIR>`  
700: executable file, only owner has permission  
740: executable file, only user in group has permission, only owner can edit  
664: normal file, user in group can read and edit, other read only  
755: executable file, only owner can edit, other can read and execute  
for directory, x means can be accessed using `cd`  
BE CAREFUL TO GIVE W PERMISSION (can edit and delete)

## file type
- regular file (-): ascii file, executable file, data file
- directory (d)
- link (l)
- device - block (b): block device, like disk, `/dev/sda`  
- device - character (c): character device, like mouse, keyboard and monitor `/dev/video`
- sockets (s): usually under `/run` or `/tmp` 
- fifo pipe (p): used to solve he problem that a file be read by multiple programs

## limitations
length: file name / dir name should be less than 255 bytes (255 ascii char)    
better to avoid some characters: `?><;&![]|\'"{}`  


## FHS (Filesystem Hierarchy Standard)
- /usr: static, sharable, dir to put software (unix software resource)
- /etc: static, unsharable, dir to put configurations
- /opt: static, sharable, dir to put 3p software
- /boot: static, unsharable, boot related
- /var/mail: variable, sharable, user email
- /var/run: variable, unsharable, programs related
`/var` is usually related with system runtime

## tmp dir
files in /tmp are managed by OS, can be deleted after reboot  
files in /var/tmp are managed by user programs, cannot be deleted after reboot

## find
find `<path>` -name `<pattern>`: find the files in the specified path that match the specified pattern  
find `<path>` -type `<type>`: find the files in the specified path that match the specified type  
find `<path>` -size `<size>`: find the files in the specified path that match the specified size  
find -exec `<command>` {}: execute the specified command for each file found
