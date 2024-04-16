#!/usr/bin/python
import os
import time

now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def main():
    res = os.popen('ifconfig | grep wlan0').read()
    if 'wlan0' not in res:
        print(f'[{now}] wifi is down, restart...')
        os.system('sudo ip link set wlan0 up')
    else:
        print(f'[{now} wifi is up]')

if __name__ == '__main__':
    main()