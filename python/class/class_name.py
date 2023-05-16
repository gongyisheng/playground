class TestClass(object):
    def __init__(self):
        print(self.__class__.__name__)

if __name__ == "__main__":
    t = TestClass()