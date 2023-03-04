# Route Table
### What is a route table?
a database that contains information about the network topology and how to reach different networks and hosts.  
### How to understand the route table?
`netstat -r` will display the route table in the following format:
```
Destination  Gateway  Flags  Refs  Use  Netif Expire
```
* Destination: the destination network or host
* Gateway: the next hop to reach the destination
* Flags: the route flags
* Refs: the number of references to this route
* Use: the number of times this route has been used
* Netif: the interface to use to reach the destination
* Expire: the expiration time of this route

1. If the destination IP address matches a network address in the routing table, the device forwards the packet to the next-hop address or interface specified in the table.   
2. If there is no matching network in the routing table, the device may send the packet to a default gateway or drop the packet if no default gateway is defined.
### How is the route table updated?
The routing table is dynamic and can be updated through routing protocols, such as `OSPF` or `BGP`, which allow routers to exchange routing information and update their routing tables based on changes in the network topology.
### Flags
- U (up): The route is up and available for use.
- G (gateway): The route is a gateway route, which means that packets need to be forwarded to a gateway instead of being delivered directly to the destination network or host.
- H (host): The route is a host route, which means that the destination is a specific host instead of a network.
- D (dynamic): The route was learned dynamically through a routing protocol, such as OSPF or BGP.
- M (modified): The route was modified by a routing protocol.
- R (reject): The route is a reject route, which means that packets destined for the route are dropped and an ICMP message is sent to the source.
- C (Cloning): A new route is cloned from this entry when it is used
- L (Link): Link-level information, such as the Ethernet MAC address, is present
- S (Static): Route added with the `route` command
- W (WasCloned): The route is a clone of another route.
- I (Installed): The route is installed in the routing table.
- c (cache): The route was added to the routing table dynamically through a routing protocol, but it's currently being cached because the next-hop address is not yet known.
- g (Good): The route was validated as a good route.
- i (IfScope): The scope of the route is limited to a specific interface.
- r (RejecT): The route is a reject route, which means that packets destined for the route are dropped and an ICMP message is sent to the source.
eg. default, UGScg; localhost, UH; 127, UCS
### Netif
Network Interface. It indicates the network interface that is used to forward packets to the destination network.
- en0: This is typically the name of the first Ethernet interface on a macOS system.
- eth0: This is typically the name of the first Ethernet interface on a Linux or Unix system.
- wlan0: This is typically the name of the first Wi-Fi interface on a Linux or Unix system.
- lo: This is typically the name of the local loopback interface, which is used to send packets to the same system without going through the network.
- utun0: This is typically the name of "User Tunnel" and is used to create virtual network interfaces that can be used for VPN connections or other types of tunneling.
### How to add/delete/update a route to the route table?
* add: `route add -net`
* delete: `route delete -net`
* update: `route change -net`