`ip.addr == 192.168.1.1` 过滤 SRC IP 或 DST IP 是 192.168.1.1 的包  
`ip.dst == 192.168.1.1` 过滤 DST IP 是 192.168.1.1 的包
`ip.src == 192.168.1.1` 过滤 SRC IP 是 192.168.1.1 的包

`tcp.port == 80` 过滤 TCP 端口是 80 的包
`tcp.flags.reset == 1` 过滤 TCP RST 包。先找到 RST 包，然后右键 Follow -> TCP Stream 是常用的排障方式
`tcp.analysis.retransmission` 过滤所有的重传包