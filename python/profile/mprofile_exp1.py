from pympler import muppy
from pympler import summary
from pympler import refbrowser
from recursive_getsizeof import total_size

class TestClass(object):
    def __init__(self):
        self.a = [1,2,3]*1000
        self.b = [4,5,6]*1000
        self.description = "This is a test class"*1000
    def func(self):
        self.c = self.a + self.b
        return self.c

def output_function(obj):
    return str(type(obj))

def main():
    all_objects = muppy.get_objects()
    obj_num1 = len(all_objects)
    sum1 = summary.summarize(all_objects)

    obj = TestClass()
    _test = obj.func()
    
    all_objects = muppy.get_objects()
    obj_num2 = len(all_objects)
    sum2 = summary.summarize(all_objects)
    diff = summary.get_diff(sum1, sum2)

    print("object number diff: ", obj_num2 - obj_num1)
    summary.print_(diff)

    # get all objects of a certain type (usually a self-defined class)
    my_types = muppy.filter(all_objects, Type=TestClass)
    print("-----Test class objects memory usage-----")
    for o in my_types:
        print(o, total_size(o))
    
    # see the object tree
    print("-----Object tree-----")
    root_ref = obj
    cb = refbrowser.ConsoleBrowser(root_ref, maxdepth=3, str_func=output_function)
    cb.print_tree()

    # get first level referents and their sizes
    print("-----First level referents-----")
    refs = muppy.get_referents(root_ref, level=1)[0]
    for k,v in refs.items():
        print(k, total_size(v)/8)

if __name__ == '__main__':
    main()


# Output:
"""
object number diff:  8291
                               types |   # objects |   total size
==================================== | =========== | ============
                                list |        3734 |    811.62 KB
                                 str |        3728 |    278.64 KB
                                 int |         823 |     22.51 KB
                               tuple |           3 |    168     B
                                dict |           1 |    104     B
                             weakref |           1 |     72     B
                                code |           0 |     70     B
                  __main__.TestClass |           1 |     48     B
                   function (setter) |           0 |      0     B
                function (_missing_) |           0 |      0     B
                 function (encoding) |           0 |      0     B
                     function (triu) |           0 |      0     B
                   function (anchor) |           0 |      0     B
  function (no_type_check_decorator) |           0 |      0     B
  function (_get_instructions_bytes) |           0 |      0     B
-----Test class objects memory usage-----
<__main__.TestClass object at 0x10139a340> 48
-----Object tree-----
<class '__main__.TestClass'>-+-<class 'list'>
                             +-<class 'list'>
-----First level referents-----
a 3017.5
b 3017.5
description 2506.125
c 6028.0
"""