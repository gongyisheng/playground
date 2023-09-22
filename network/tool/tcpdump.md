### save tcpdump result to file  
- interface: eth0
- host: <>
- snaplen: 96
- output file: /tmp/tcpdump.pcap  
`sudo tcpdump -i eth0 host <> -w /tmp/tcpdump.pcap -s 120 -C 5m`

ref: https://www.tcpdump.org/manpages/tcpdump.1.html