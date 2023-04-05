### save tcpdump result to file  
- interface: eth0
- host: <>
- snaplen: 96
- output file: /tmp/tcpdump.pcap  
`sudo tcpdump -i eth0 host <> -w /tmp/tcpdump.pcap -s 96`

ref: https://www.tcpdump.org/manpages/tcpdump.1.html