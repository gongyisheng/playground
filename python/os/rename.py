import os
import time

def main():
    f = open('test.txt', 'w')
    f.write('Hello, world!')
    f.close()

    time.sleep(2)

    f = open('test2.txt', 'w')
    f.write('Hello, world!')
    f.close()

    time.sleep(2)

    os.rename('test2.txt', 'test.txt')

    time.sleep(2)

    os.remove('test.txt')

if __name__ == '__main__':
    main()