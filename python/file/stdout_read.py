import sys


def main():
    print("Hello, world!", file=open("stdout_read.txt", "w"))
    print(sys.stdout.fileno())
    print(sys.stdout.isatty())
    print(sys.stdout.name)


if __name__ == "__main__":
    main()
