### Problem:
How to allow objects with incompatible interfaces to collaborate.
### Solution:
Use an adapter to convert the interface of one object so that another object can understand it.
### When to use:
Conversion between interfaces.
### Steps:
1. The adapter gets an interface, compatible with one of the existing objects.
2. Using this interface, the existing object can safely call the adapter’s methods.
3. Upon receiving a call, the adapter passes the request to the second object, but in a format and order that the second object expects.
### Example:
```
class Target(object):
    def request(self):
        return "Target: The default target's behavior."

class Adaptee(object):
    def specific_request(self):
        return ".eetpadA eht fo roivaheb laicepS"

class Adapter(Target):
    def __init__(self, adaptee):
        self._adaptee = adaptee
    
    def _translate(self):
        # customized translate logic
        return self._adaptee.specific_request()[::-1]

    def request(self):
        return "Adapter: (TRANSLATED) {}".format(self._translate())

if __name__ == "__main__":
    adaptee = Adaptee()

    # works well with Adapter
    Target().request()
    Adapter(adaptee).request()
```