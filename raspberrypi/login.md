# change password
change password of current user:  
`passwd`  
create a new user:  
`sudo useradd -m <username> -G sudo`  
`sudo passwd <username>`  
delete default pi user:  
`sudo deluser pi`  
delete user with its home folder  
`sudo deluser -remove-home pi`  