# nc - netcat
## check connectivity
send tcp handshake to the address/ip + port to check connectivity
```
# zero I/O mode, just scan
nc -zv google.com 80
nc -zv google.com 443
nc -zv 1.1.1.1 53

# tcp
nc -tv google.com 80

# udp
nc -uv 1.1.1.1 53

```

## port scan
```
nc -zv 127.0.0.1 1-65535
```

## send tcp packets with raw payload
```
# check http response
echo -e "GET / HTTP/1.1\r\nHost: <hostname>\r\nConnection: close\r\n\r\n" | nc google.com 80

# check dns response
echo -n -e "\x13\x37\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x13google-public-dns-a\x06google\x03com\x00\x00\x01\x00\x01" | nc -u -w1 8.8.8.8 53 | hexdump -C
echo -n -e "\x3f\x85\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x04time\x05apple\x03com\x00\x00\x01\x00\x01" | nc -u -w1 8.8.8.8 53 | hexdump -C

note:
\x3f\x85 - Flags for standard, recursive, 1-question query
\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00 - Flags for standard, recursive, 1-question query
\x04time\x05apple\x03com\x00 - Query string, null-terminated, each section starts with length number
\x00\x01\x00\x01 - Query type (A) and class (IN)
```

## send files
```
# receiver
nc -l 12345 < filename
# sender
nc <sender_ip> 12345 > output_filename
```
