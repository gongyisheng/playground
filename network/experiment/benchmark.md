Experiment benchmark settings
1. Start 2 EC2 instance on AWS in the same region, one as server, one as client.
2. Edit security group of server, add inbound rule: `Custom TCP Rule`, `Port Range: 8000`
3. Start server: `python -m http.server 8000`
4. Client can get files through `curl`: `curl http://172.31.82.1:8080/0000000000000000.data --output ~/0000000000000000.data`
   
Download file:  
`scp -i ~/.ssh/yipit-mac.pem ec2-user@*.*.*.*:~/client.pcap ~/Downloads`  
`scp -i ~/.ssh/yipit-mac.pem ec2-user@*.*.*.*:~/server.pcap ~/Downloads`

### install tools
refer to: https://command-not-found.com/

### traffic control
give packets from enX0 a delay of 2ms (client)
`tc qdisc add dev enX0 root netem delay 2ms`
give packets from enX0 a delay of 100ms (client)
`tc qdisc change dev enX0 root netem delay 100ms`
give packets from enX0 a loss rate of 1% (server)
`tc qdisc add dev enX0 root netem loss 1%`
give packets from enX0 a loss rate of 20% (server)
`tc qdisc change dev enX0 root netem loss 20%`
give packets from enX0 a duplicate rate of 1% (server)
`tc qdisc add dev enX0 root netem duplicate 1%`
give packets from enX0 a duplicate rate of 20% (server)
`tc qdisc change dev enX0 root netem duplicate 20%`
give packets from enX0 a corrupt rate of 1% (server)
`tc qdisc add dev enX0 root netem corrupt 1%`
give packets from enX0 a corrupt rate of 20% (server)
`tc qdisc change dev enX0 root netem corrupt 20%`
give packets from enX0 a reorder rate of 1% (server)
`tc qdisc add dev enX0 root netem delay 100ms reorder 99% 10%`
give packets from enX0 a reorder rate of 20% (server)
`tc qdisc change dev enX0 root netem delay 100ms reorder 80% 10%`
give packets from enX0 a speed limit of 50Mbps (server)
`tc qdisc add dev enX0 root tbf rate 400mbit burst 10mbit latency 10ms`
give packets from enX0 a speed limit of 1Mbps (server)
`tc qdisc change dev enX0 root tbf rate 8mbit burst 200kbit latency 10ms`

delete all traffic control rules
`tc qdisc del dev enX0 root`

## buffer size
see current buffer size settings:
`sysctl -a|egrep "rmem|wmem"`

origin settings:
`sysctl -w net.ipv4.tcp_rmem="4096 131072 6291456"`
`sysctl -w net.ipv4.tcp_wmem="4096 20480 4194304`

hardcode read buffer size to 4096 bits (client)
`sysctl -w net.ipv4.tcp_rmem="4096 4096 4096"`
hardcode write buffer size to 4096 bits (server)
`sysctl -w net.ipv4.tcp_wmem="4096 4096 4096"`

## buffer experiment settings
#### wmem
- wmem_4096_rtt_100ms: server wmem=4096, client delay=100ms
- wmem_4096_loss_20%: server wmem=4096, server loss=20%
- wmem_4096_duplicate_20%: server wmem=4096, server duplicate=20%
- wmen_4096_corrupt_20%: server wmem=4096, server corrupt=20%
- wmen_4096_reorder_20%: server wmem=4096, server reorder=20%
- wmen_4096_bandwidth_1mbps: server wmem=4096, server speed=1Mbps
#### rmen
- rmen_4096_rtt_100ms: client rmem=4096, client delay=100ms
- rmen_4096_loss_20%: client rmem=4096, server loss=20%
- rmen_4096_duplicate_20%: client rmem=4096, server duplicate=20%
- rmen_4096_corrupt_20%: client rmem=4096, server corrupt=20%
- rmen_4096_reorder_20%: client rmem=4096, server reorder=20%
- rmen_4096_bandwidth_1mbps: client rmem=4096, server speed=1Mbps
