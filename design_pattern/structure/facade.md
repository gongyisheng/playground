### Problem:
How to decouple your code from the underlying 3p-libraries and frameworks.
### Solution:
Create a simple interface to a complex subsystem with limited functionality.
### When to use:
Hide/Decouple the complexity of a subsystem from the client.
### Steps:
1. Dive deep to understand the subsystem(s) you use.
2. Create a Facade class that provides a simple interface to complex subsystem(s).
3. Put all operational steps in the Facade class.
### Example:
```
class Facade:

    def __init__(self, subsystem1: Subsystem1, subsystem2: Subsystem2) -> None:

        self._subsystem1 = subsystem1 or Subsystem1()
        self._subsystem2 = subsystem2 or Subsystem2()

    def operation(self) -> str:

        results = []
        results.append("Facade initializes subsystems:")
        results.append(self._subsystem1.operation1())
        results.append(self._subsystem2.operation1())
        results.append("Facade orders subsystems to perform the action:")
        results.append(self._subsystem1.operation_n())
        results.append(self._subsystem2.operation_z())
        return "\n".join(results)


class Subsystem1:

    def operation1(self) -> str:
        return "Subsystem1: Ready!"

    # ...

    def operation_n(self) -> str:
        return "Subsystem1: Go!"


class Subsystem2:

    def operation1(self) -> str:
        return "Subsystem2: Get ready!"

    # ...

    def operation_z(self) -> str:
        return "Subsystem2: Fire!"


def client_code(facade: Facade) -> None:
    print(facade.operation(), end="")


if __name__ == "__main__":
    subsystem1 = Subsystem1()
    subsystem2 = Subsystem2()
    facade = Facade(subsystem1, subsystem2)
    client_code(facade)
```