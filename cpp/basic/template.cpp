#include <iostream>
#include <string>

// Templates in C++ are mostly used for generic programming.

// We start with the kind of generic programming you might be familiar with. 
// To define a class or function that takes a type parameter:
template<class T>

// a template class
class Box {
    public:
        // In this class, T can be used as any other type.
        void insert(const T&) { /* ... */ }
};

// a template function
// Template parameters don't have to be classes
template<int I>
void printMessage() {
    std::cout << "message: " << I << std::endl;
}

int main() {
    // To instantiate a template class on the stack:
    Box<int> intBox;

    // and you can use it as you would expect:
    intBox.insert(123);

    // You can, of course, nest templates:
    Box<Box<int>> boxOfBox;
    boxOfBox.insert(intBox);

    // To execute a template function:
    printMessage<123>();
}