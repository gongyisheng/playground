import random


def run_inf_loop():
    a = 0
    while True:
        rand = random.randint(0, 1000)
        if rand % 2 == 0:
            a += rand
        else:
            a -= rand
        if a > 1_000_000_000:
            a = 0


if __name__ == "__main__":
    run_inf_loop()
