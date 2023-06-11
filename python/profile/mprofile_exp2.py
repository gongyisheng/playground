from pympler import classtracker, asizeof
import time

class TestClass(object):
    def __init__(self):
        self.a = [1,2,3]*1000
        self.b = [4,5,6]*1000
        self.description = "This is a test class"*1000
    def func(self):
        self.c = self.a + self.b
        return self.c

def main():
    # class tracker, track all instances of a class, periodically take snapshots
    class_tr = classtracker.ClassTracker()
    class_tr.track_class(TestClass)
    class_tr.start_periodic_snapshots(1)

    obj = TestClass()

    # track object size, track a specific object, periodically take snapshots
    obj_tr = classtracker.ClassTracker()
    obj_tr.track_object(obj)
    obj_tr.start_periodic_snapshots(1)

    # track object size, manually take snapshots
    tracked_obj = classtracker.TrackedObject(obj, "TestClass")
    
    _test1 = obj.func()
    tracked_obj.track_size(time.time(), asizeof.Asizer()) # manually take a snapshot
    obj._desc1 = "This is a test class"*1000
    tracked_obj.track_size(time.time(), asizeof.Asizer()) # manually take a snapshot
    time.sleep(2)
    obj._desc2 = "This is a test class"*1000
    tracked_obj.track_size(time.time(), asizeof.Asizer()) # manually take a snapshot
    
    time.sleep(2)

    _obj = TestClass()
    _test2 = _obj.func()
    time.sleep(2)
    _obj._desc1 = "Hello"*10000
    time.sleep(2)
    
    print("-----Test object memory usage-----")
    obj_tr.stats.print_summary()
    obj_tr.stats.print_object(tracked_obj)

    print("-----Test class memory usage-----")
    class_tr.stats.print_summary()

    return (_test1, _test2)

if __name__ == '__main__':
    main()

# Output:
"""
-----Test object memory usage-----
---- SUMMARY ------------------------------------------------------------------
                                         active      0     B      average   pct
  TestClass                                   1    133.70 KB    133.70 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    133.70 KB    133.70 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  TestClass                                   1    153.38 KB    153.38 KB    0%
-------------------------------------------------------------------------------
TestClass                        0x104aa1a00 <__main__.TestClass object at 0x...
  00:00:00.00                      48     B
  468465:54:46.55                 114.06 KB
  468465:54:46.55                 133.70 KB
  468465:54:48.56                 153.38 KB
-----Test class memory usage-----
---- SUMMARY ------------------------------------------------------------------
                                         active      0     B      average   pct
  __main__.TestClass                          0      0     B      0     B    0%
                                         active      0     B      average   pct
  __main__.TestClass                          1    133.70 KB    133.70 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          1    153.38 KB    153.38 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          2    267.07 KB    133.54 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          2    267.07 KB    133.54 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          2    315.95 KB    157.98 KB    0%
                                         active      0     B      average   pct
  __main__.TestClass                          2    315.95 KB    157.98 KB    0%
-------------------------------------------------------------------------------
"""