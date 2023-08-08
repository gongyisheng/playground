import sys, select, tty, termios
import time

old_attr = termios.tcgetattr(sys.stdin) 
tty.setcbreak(sys.stdin.fileno())   
print('Please input keys, press Ctrl + C to quit')
f = open('event.txt', 'w')

while(1):
    if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
        f.write(f"{sys.stdin.read(1)},{time.time()}\n")

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)