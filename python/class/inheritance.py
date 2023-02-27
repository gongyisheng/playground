class parent(object):
    VAR1 = "Parent class variable VAR1"
    VAR2 = "Parent class variable VAR2"
    
    def __init__(self, name):
        self.name = name
        print("Parent class constructor called")

class child1(parent):
    
    def __init__(self, name):
        super(child1, self).__init__(name)
        print("Child1 class constructor called")
        self.VAR1 = "Child1 class variable VAR1"
        print(self.VAR1)

class child2(parent):
        
        def __init__(self, name):
            super(child2, self).__init__(name)
            print("Child2 class constructor called")
            self.VAR2 = "Child2 class variable VAR2"
            print(self.VAR2)

def test():
    p = parent("Parent")
    c1 = child1("Child1")
    c2 = child2("Child2")
    print("-----------------")
    print("Parent class variable VAR1: ", p.VAR1)
    print("Parent class variable VAR2: ", p.VAR2)
    print("-----------------")
    print("Child1 class variable VAR1: ", c1.VAR1)
    print("Child1 class variable VAR2: ", c1.VAR2)
    print("-----------------")
    print("Child2 class variable VAR1: ", c2.VAR1)
    print("Child2 class variable VAR2: ", c2.VAR2)
    print("-----------------")

if __name__ == "__main__":
    test()