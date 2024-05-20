def follow(f, target):
    f.seek(0, 2)
    while True:
        last_line = f.readline()
        if last_line is not None:
            target.send(last_line)


def printer():
    while True:
        line = yield
        print(line, end="")


def main():
    f = open("channel-data.txt")
    prt = printer()
    next(prt)  # initialize the couroutine
    follow(f, prt)  # start the event loop


if __name__ == "__main__":
    main()
