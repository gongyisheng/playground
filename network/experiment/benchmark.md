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
`tc qdisc add dev enX0 root netem delay 10ms reorder 99% 50%`
give packets from enX0 a reorder rate of 20% (server)
`tc qdisc change dev enX0 root netem delay 10ms reorder 80% 50%`