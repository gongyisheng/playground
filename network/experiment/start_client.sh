rm -f 0000000000000000.data client.pcap
ping baidu.com -c 1
sleep 2
curl http://172.31.82.1:8080/0000000000000000.data --output ~/0000000000000000.data
ping baidu.com -c 1