import pytest

@pytest.fixture
def setup_and_teardown():
    # setup code
    print("setup")
    
    yield

    # teardown code
    print("teardown")

def test_something(setup_and_teardown):
    # test code
    print("running test_something")