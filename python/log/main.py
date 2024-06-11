import logging
from logger import setup_logger
from utils import print_something


def main():
    setup_logger()
    print_something()
    logging.info("Hello World")


if __name__ == "__main__":
    main()
    # Output:
    # INFO: [2023-12-01 14:10:23,731][root][utils][utils.py:3]something
    # INFO: [2023-12-01 14:10:23,731][root][main][main.py:8]Hello World
