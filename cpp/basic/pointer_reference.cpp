// Differences between References and Pointers in C++
//
// ┌─────────────────────┬──────────────────────────────┬───────────────────────┐
// │       Feature       │        Reference (&)         │      Pointer (*)      │
// ├─────────────────────┼──────────────────────────────┼───────────────────────┤
// │ Must be initialized │ Yes                          │ No (can be null)      │
// │ Can be null         │ No                           │ Yes                   │
// │ Can be reassigned   │ No                           │ Yes                   │
// │ Syntax              │ Use directly like the object │ Need * to dereference │
// │ Memory address      │ Hidden                       │ Explicit with &       │
// └─────────────────────┴──────────────────────────────┴───────────────────────┘

#include <iostream>

// 1. Declaration & Initialization
void declaration_example() {
    std::cout << "=== Declaration & Initialization ===\n";

    int x = 10;

    // Reference - MUST be initialized, becomes alias for x
    int& ref = x;      // ref IS x now
    // int& ref2;      // ERROR: references must be initialized

    // Pointer - can be null or uninitialized
    int* ptr = &x;     // ptr HOLDS ADDRESS of x
    int* ptr2 = nullptr;  // OK: null pointer

    std::cout << "x = " << x << ", ref = " << ref << ", *ptr = " << *ptr << "\n";
    std::cout << "ptr2 is " << (ptr2 == nullptr ? "null" : "not null") << "\n\n";
}

// 2. Usage Syntax
void usage_syntax_example() {
    std::cout << "=== Usage Syntax ===\n";

    int x = 10;
    int& ref = x;
    int* ptr = &x;

    std::cout << "Initial x = " << x << "\n";

    // Reference: use directly
    ref = 20;          // x is now 20
    std::cout << "After ref = 20: x = " << x << ", ref = " << ref << "\n";

    // Pointer: must dereference with *
    *ptr = 30;         // x is now 30
    std::cout << "After *ptr = 30: x = " << x << ", *ptr = " << *ptr << "\n\n";
}

// 3. Reassignment behavior
void reassignment_example() {
    std::cout << "=== Reassignment ===\n";

    int a = 1, b = 2;

    // Reference: cannot rebind
    int& ref = a;
    std::cout << "Before: a = " << a << ", b = " << b << ", ref = " << ref << "\n";

    ref = b;           // This assigns b's VALUE to a, doesn't rebind ref!
                       // Now a == 2, ref still refers to a
    std::cout << "After ref = b: a = " << a << ", b = " << b << ", ref = " << ref << "\n";
    std::cout << "ref still refers to a (both are 2 now)\n\n";

    // Pointer: can point to different things
    int c = 100, d = 200;
    int* ptr = &c;
    std::cout << "Before: c = " << c << ", d = " << d << ", *ptr = " << *ptr << "\n";

    ptr = &d;          // Now ptr points to d
    std::cout << "After ptr = &d: *ptr = " << *ptr << " (now points to d)\n\n";
}

// 4. Null Safety - Reference version (guaranteed to be valid)
void process_ref(int& ref) {
    ref = 100;     // Always safe - no null check needed
}

// 4. Null Safety - Pointer version (might be null)
void process_ptr(int* ptr) {
    if (ptr != nullptr) {  // Need to check!
        *ptr = 100;
    }
}

void null_safety_example() {
    std::cout << "=== Null Safety ===\n";

    int x = 50;

    // Reference: always valid
    process_ref(x);
    std::cout << "After process_ref: x = " << x << "\n";

    // Pointer: can pass nullptr safely
    int y = 50;
    process_ptr(&y);
    std::cout << "After process_ptr(&y): y = " << y << "\n";

    process_ptr(nullptr);  // Safe - function checks for null
    std::cout << "process_ptr(nullptr) handled safely\n\n";
}

// 5. When to use each
//
// Use REFERENCES when:
// - The value must exist (non-optional)
// - You won't need to rebind to another object
// - You want cleaner syntax
//
// Use POINTERS when:
// - The value might be null/optional
// - You need to reassign to different objects
// - You're doing manual memory management
// - Working with arrays or pointer arithmetic

// Example: Singleton returns reference (instance always exists)
class Singleton {
public:
    static Singleton& instance() {
        static Singleton s;
        return s;  // Reference: singleton always exists
    }
    void sayHello() { std::cout << "Hello from Singleton!\n"; }
private:
    Singleton() = default;
};

// Example: Output parameter uses pointer-to-pointer
bool allocate_memory(void** ptr, size_t size) {
    *ptr = malloc(size);  // Need to modify where ptr points
    return *ptr != nullptr;
}

int main() {
    declaration_example();
    usage_syntax_example();
    reassignment_example();
    null_safety_example();

    std::cout << "=== Singleton Example ===\n";
    Singleton::instance().sayHello();

    std::cout << "\n=== Output Parameter Example ===\n";
    void* buffer = nullptr;
    if (allocate_memory(&buffer, 1024)) {
        std::cout << "Allocated 1024 bytes at " << buffer << "\n";
        free(buffer);
    }

    return 0;
}
