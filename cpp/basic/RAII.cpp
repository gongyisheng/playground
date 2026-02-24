// RAII stands for "Resource Acquisition Is Initialization".
// It is often considered the most powerful paradigm in C++
// and is the simple concept that a constructor for an object
// acquires that object's resources and the destructor releases them.

// In C++, any {} creates a scope, and objects are destroyed when their scope ends. 
#include <iostream>
#include <fstream>
#include <string>

// test: g++ -o build/RAII basic/RAII.cpp && build/RAII

void doA(const std::ifstream &fh) {
    std::cout << "doA" << std::endl;
}

void doB(const std::ifstream &fh) {
    std::cout << "doB" << std::endl;
}

// fstream uses its destructor to close the file.
// Recall from above that destructors are automatically called
// whenever an object falls out of scope.
void doSomethingWithAFile(const std::string& filename)
{
    // ifstream is short for input file stream
    std::ifstream fh(filename); // Open the file

    // Do things with the file
    doA(fh);
    doB(fh);

} // The file is automatically closed here by the destructor

// This has _massive_ advantages:
// 1. No matter what happens,
//    the resource (in this case the file handle) will be cleaned up.
//    Once you write the destructor correctly,
//    It is _impossible_ to forget to close the handle and leak the resource.
// 2. Note that the code is much cleaner.
//    The destructor handles closing the file behind the scenes
//    without you having to worry about it.
// 3. The code is exception safe.
//    An exception can be thrown anywhere in the function and cleanup
//    will still occur.

// All idiomatic C++ code uses RAII extensively for all resources.
// Additional examples include
// - Memory using unique_ptr and shared_ptr
// - Containers - the standard library linked list,
//   vector (i.e. self-resizing array), hash maps, and so on
//   all automatically destroy their contents when they fall out of scope.
// - Mutexes using lock_guard and unique_lock

int main()
{
    doSomethingWithAFile("test.txt");
    return 0;
}