import logging

class MyGenIt(object):
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def __iter__(self):
        with self:
            for o in self.content:
                yield o
    def __enter__(self):
        return self
    def __exit__(self,  exc_type, exc_value, traceback):
        if exc_type:
            logging.error("Aborted %s", self,
                          exc_info=(exc_type, exc_value, traceback))
def generator():
    for i in range(10):
        yield i

if __name__ == "__main__":
    for x in MyGenIt("foo",range(10)):
        if x == 5:
            raise ValueError("got 5")