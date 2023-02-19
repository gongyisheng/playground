### Problem:
How to create an exact copy of an object even with private properties and without introducing dependencies?
### Solution:
Implement a common interface to delegate the cloning process to the actual objects that are being cloned.
### When to use:
Copy objects
### Steps:
1. Create Prototype interface declares the cloning methods.
2. Create Concrete Prototype class implements the cloning method. (Deep copy)
3. Now client can produce a copy of any object that follows the prototype interface.
### Example:
```
class Prototype(object):
    def clone(self):
        pass

class ConcretePrototype(Prototype):
    def __init__(self, name, dependent_ref):
        self.name = name
        self.dependent_ref = dependent_ref

    def __copy__(self):
        # shallow copy
        copy_of_name = copy.copy(self.name)
        copy_of_dependent_ref = copy.copy(self.dependent_ref)

        new = self.__class__(copy_of_name, copy_of_dependent_ref)
        new.__dict__.update(self.__dict__)

        return new
    
    def __deepcopy__(self):
        # deep copy
        if memo is None:
            memo = {}
        
        copy_of_name = copy.deepcopy(self.name, memo)
        copy_of_dependent_ref = copy.deepcopy(self.dependent_ref, memo)

        new = self.__class__(copy_of_name, copy_of_dependent_ref)
        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new
```