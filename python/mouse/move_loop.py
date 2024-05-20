from pynput.mouse import Button, Controller
import time
import random


def main():
    mouse = Controller()
    while True:
        time.sleep(random.randint(1, 10))
        mouse.position = (random.randint(0, 1440), random.randint(0, 900))
        print("Current position: " + str(mouse.position))


if __name__ == "__main__":
    main()
