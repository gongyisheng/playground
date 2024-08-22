import time

def a():
    l = []
    while True:
        num = int(1024*1024*1024/2)
        l.append("a"*num) #500mb
        time.sleep(10)
a()