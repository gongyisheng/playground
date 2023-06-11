from pympler import muppy
from pympler import summary

class TestClass(object):
    def __init__(self):
        self.a = [1,2,3]*1000
        self.b = [4,5,6]*1000
        self.description = "This is a test class"*1000
    def func(self):
        self.c = self.a + self.b
        return self.c

def main():
    obj = TestClass()
    _test = obj.func()
    all_objects = muppy.get_objects()
    print(len(all_objects))
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

if __name__ == '__main__':
    main()
    """
    use object number comparison to get an idea whether it's caused by undestroyed object
    use total size comparison to get an idea which type of object is causing the problem 
    """
    """
    temp@Orange-cats-shared-mac profile % python mprofile_exp.py
    46623
                           types |   # objects |   total size
    ============================ | =========== | ============
                             str |       15597 |     22.02 MB
                            dict |        4181 |      1.65 MB
                            code |        5100 |    880.77 KB
                            type |         825 |    633.25 KB
                             set |         138 |    201.61 KB
                           tuple |        3368 |    191.67 KB
                            list |         309 |    159.85 KB
              wrapper_descriptor |        2169 |    152.51 KB
              _io.BufferedWriter |           2 |    128.33 KB
      builtin_function_or_method |        1375 |     96.68 KB
                     abc.ABCMeta |          87 |     85.27 KB
               method_descriptor |        1210 |     85.08 KB
                         weakref |        1151 |     80.93 KB
              _io.BufferedReader |           1 |     64.16 KB
               getset_descriptor |         860 |     53.75 KB
    """

    """
    temp@Orange-cats-shared-mac profile % python mprofile_exp.py
    46623
                           types |   # objects |   total size
    ============================ | =========== | ============
                             str |       15597 |      4.86 MB
                            dict |        4181 |      1.65 MB
                            code |        5100 |    880.77 KB
                            type |         825 |    633.25 KB
                             set |         138 |    201.61 KB
                           tuple |        3368 |    191.67 KB
                            list |         309 |    159.85 KB
              wrapper_descriptor |        2169 |    152.51 KB
              _io.BufferedWriter |           2 |    128.33 KB
      builtin_function_or_method |        1375 |     96.68 KB
                     abc.ABCMeta |          87 |     85.27 KB
               method_descriptor |        1210 |     85.08 KB
                         weakref |        1151 |     80.93 KB
              _io.BufferedReader |           1 |     64.16 KB
               getset_descriptor |         860 |     53.75 KB
    """