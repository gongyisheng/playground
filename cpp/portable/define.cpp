#include <iostream>

// test: g++ -o build/define portable/define.cpp && build/define

// Simple constant
#define MAX_SIZE 100

// Function-like macro
// Parentheses around arguments prevent operator precedence issues
// e.g., SQUARE(1+2) -> ((1+2)*(1+2)) = 9, not 1+2*1+2 = 5
#define SQUARE(x) ((x) * (x))

// Multi-statement macro using do-while(false)
// Why do { } while(false)?
// - Makes macro behave as a single statement
// - Requires semicolon after use
// - Safe with if-else: if (x) MACRO(); else foo(); works correctly
// Without it: if (x) { a; b; }; else foo(); breaks (dangling else)
#define PRINT_TWICE(msg) \
    do { \
        std::cout << msg << std::endl; \
        std::cout << msg << std::endl; \
    } while (false)

// Stringification (#) - converts argument to string literal
#define TO_STRING(x) #x

// Token pasting (##) - concatenates tokens
#define MAKE_VAR(name) var_##name

// Variadic macro (__VA_ARGS__)
// ... captures any number of additional arguments
// __VA_ARGS__ expands to those captured arguments
// ## before __VA_ARGS__ removes trailing comma when no args provided
//   - LOG("hi")      -> printf("hi\n")        (## removes comma)
//   - LOG("x=%d", 1) -> printf("x=%d\n", 1)   (## does nothing)
#define LOG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// Conditional compilation
// Switch debug on/off:
//   - Comment/uncomment: #define DEBUG_MODE
//   - Compiler flag: g++ -DDEBUG_MODE main.cpp (no need to define in code)
//   - Use #undef DEBUG_MODE to turn off later
#define DEBUG_MODE

int main() {
    // Simple constant
    std::cout << "=== Simple Constant ===" << std::endl;
    std::cout << "MAX_SIZE = " << MAX_SIZE << std::endl;

    // Function-like macro
    std::cout << "\n=== Function-like Macro ===" << std::endl;
    std::cout << "SQUARE(5) = " << SQUARE(5) << std::endl;
    std::cout << "SQUARE(1+2) = " << SQUARE(1+2) << std::endl;

    // Multi-statement macro
    std::cout << "\n=== Multi-statement Macro ===" << std::endl;
    PRINT_TWICE("Hello");

    // Stringification
    std::cout << "\n=== Stringification (#) ===" << std::endl;
    std::cout << TO_STRING(hello world) << std::endl;
    int myVar = 42;
    std::cout << TO_STRING(myVar) << " = " << myVar << std::endl;

    // Token pasting
    std::cout << "\n=== Token Pasting (##) ===" << std::endl;
    int MAKE_VAR(test) = 123;  // creates: int var_test = 123;
    std::cout << "var_test = " << var_test << std::endl;

    // Variadic macro
    std::cout << "\n=== Variadic Macro ===" << std::endl;
    LOG("Value: %d", 42);
    LOG("No args");

    // Conditional compilation
    std::cout << "\n=== Conditional Compilation ===" << std::endl;
#ifdef DEBUG_MODE
    std::cout << "Debug mode enabled" << std::endl;
#else
    std::cout << "Release mode" << std::endl;
#endif

    return 0;
}
