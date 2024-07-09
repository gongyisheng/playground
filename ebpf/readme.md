# install bcc
`echo deb http://cloudfront.debian.net/debian sid main >> /etc/apt/sources.list`  
`sudo apt-get install -y bpfcc-tools libbpfcc libbpfcc-dev linux-headers-$(uname -r)`  
ref: https://github.com/iovisor/bcc/blob/master/INSTALL.md#debian---binary  