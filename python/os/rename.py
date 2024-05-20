import os
import time


def main():
    f = open("test.txt", "w")
    f.write("Hello, world!\n")
    f.write("Hello, world!\n")
    f.close()

    time.sleep(2)

    f = open("test2.txt", "w")
    f.write("Hello, world2!\n")
    f.write("Hello, world2!\n")
    f.close()

    time.sleep(2)

    os.rename("test2.txt", "test.txt")

    time.sleep(2)


if __name__ == "__main__":
    main()
