def add(x, y):
    return x + y


def test_addition():
    assert add(2, 3) == 5


def test_addition_negative():
    assert add(-1, 1) == 0


# run command
# pytest -v -s test_example.py
# pytest -k test_addition_negative
