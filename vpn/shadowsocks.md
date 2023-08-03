# install shadowsocks on x86_64 aws ec2
- sudo yum install python
- sudo yum install python-pip
- sudo pip install shadowsocks
- sudo vim /etc/shadowsocks.json  
{  
    "server":"0.0.0.0",  
    "server_port": YOUR PORT,  
    "password": YOUR PASSWORD,  
    "timeout":300,  
    "method":"aes-256-cfb",  
    "fast_open":true,  
    "workers": 1  
}  
- edit /usr/local/lib/python3.9/site-packages/shadowsocks/crypto/openssl.py, replace EVP_CIPHER_CTX_cleanup with EVP_CIPHER_CTX_reset
- sudo ssserver -c /etc/shadowsocks.json -d start
