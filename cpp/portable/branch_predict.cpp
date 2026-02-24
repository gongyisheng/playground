#include <iostream>

// test: g++ -o build/branch_predict portable/branch_predict.cpp && build/branch_predict

// Compiler-specific branch prediction hints
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#define HAS_BUILTIN_EXPECT 1
#else
#define LIKELY(expr) (expr)
#define UNLIKELY(expr) (expr)
#define HAS_BUILTIN_EXPECT 0
#endif

int main() {
#if HAS_BUILTIN_EXPECT
    std::cout << "__builtin_expect: supported" << std::endl;
#else
    std::cout << "__builtin_expect: not supported" << std::endl;
#endif

    int value = 42;
    if (LIKELY(value > 0)) {
        std::cout << "Common path (LIKELY)" << std::endl;
    }
    if (UNLIKELY(value > 100)) {
        std::cout << "Rare path (UNLIKELY)" << std::endl;
    }

    return 0;
}
