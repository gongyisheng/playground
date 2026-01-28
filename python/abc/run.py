"""
Python ABC (Abstract Base Class) Module Examples

The abc module provides infrastructure for defining abstract base classes.
Abstract classes cannot be instantiated directly and enforce that subclasses
implement certain methods.
"""

from abc import ABC, abstractmethod


# example 1: basic abstract base class

class Animal(ABC):
    """
    Abstract base class representing an animal.
    Any subclass must implement the 'speak' method.
    """

    @abstractmethod
    def speak(self) -> str:
        """Abstract method - subclasses must override this."""
        pass

    def breathe(self) -> str:
        """Concrete method - shared by all subclasses."""
        return "Breathing..."


class Dog(Animal):
    """Concrete implementation of Animal."""

    def speak(self) -> str:
        return "Woof!"


class Cat(Animal):
    """Another concrete implementation of Animal."""

    def speak(self) -> str:
        return "Meow!"


if __name__ == "__main__":
    # run example
    dog = Dog()
    cat = Cat()
    print(f"Dog says: {dog.speak()}")
    print(f"Cat says: {cat.speak()}")
    print(f"Dog breathing: {dog.breathe()}")

    # Uncommenting the line below would raise TypeError:
    # animal = Animal()  # Cannot instantiate abstract class
