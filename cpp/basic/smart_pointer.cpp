// Generally a smart pointer is a class which wraps a "raw pointer" (usage of "new"
// respectively malloc/calloc in C). The goal is to be able to
// manage the lifetime of the object being pointed to without 
// ever needing to explicitly delete the object. 
// The term itself simply describes a set of pointers with the mentioned abstraction.
// Smart pointers should preferred over raw pointers, to prevent
// risky memory leaks, which happen if you forget to delete an object.
#include <iostream>
#include <memory>

class Dog {
public:
    Dog() { std::cout << "Dog is created" << std::endl; }
    ~Dog() { std::cout << "Dog is destroyed" << std::endl; }
    void bark() { std::cout << "Woof!" << std::endl; }
};

void foo(){
    // It's no longer necessary to delete the Dog.
    std::shared_ptr<Dog> doggo(new Dog());
    doggo->bark();
}

int main() {
    // Usage of a raw pointer:
    Dog* ptr = new Dog();
    ptr->bark();
    delete ptr;

    // By using a smart pointer, you don't have to worry about 
    // the deletion of the object anymore.
    // A smart pointer describes a policy, to count the references to the
    // pointer. The object gets destroyed when the last
    // reference to the object gets destroyed.

    // Usage of "std::shared_ptr":
    foo();
    
    // Beware of possible circular references!!!
    // There will be always a reference, so it will be never destroyed!
    std::shared_ptr<Dog> doggo_one(new Dog());
    std::shared_ptr<Dog> doggo_two(new Dog());
    doggo_one = doggo_two; // p1 references p2
    doggo_two = doggo_one; // p2 references p1
}