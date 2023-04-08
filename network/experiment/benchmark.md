Experiment benchmark settings
1. Start 2 EC2 instance on AWS in the same region, one as server, one as client.
2. Edit security group of server, add inbound rule: `Custom TCP Rule`, `Port Range: 8089`
3. Start server: `python -m http.server 8089`
4. Client can get files through `curl`: `curl http://<ip>:8089/<file_path> --output ~/<file_name>`
   
Download file:  
`scp -i ~/.ssh/yipit-mac.pem ec2-user@*.*.*.*:~/client.pcap ~/Downloads`  
`scp -i ~/.ssh/yipit-mac.pem ec2-user@*.*.*.*:~/server.pcap ~/Downloads`

### install tools
refer to: https://command-not-found.com/

### traffic control
give packets from enX0 a delay of 2ms
`tc qdisc add dev enX0 root netem delay 2ms`
give packets from enX0 a delay of 500ms
`tc qdisc change dev enX0 root netem delay 500ms`