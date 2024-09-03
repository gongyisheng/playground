import memray

class TestClass(object):
    def __init__(self):
        self.a = [1, 2, 3] * 1000
        self.b = [4, 5, 6] * 1000
        self.description = "This is a test class" * 1000

    def func(self):
        self.c = self.a + self.b
        return self.c

if __name__ == "__main__":
    with memray.Tracker("memoryprof.bin", file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
        obj = TestClass()
        obj.func()
