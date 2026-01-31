#include <iostream>

// test: g++ -o build/platform portable/platform.cpp && build/platform

int main() {
    // Show which compiler is detected
    std::cout << "=== Compiler Detection ===" << std::endl;

#if defined(__clang__)
    std::cout << "Compiler: Clang " << __clang_major__ << "." << __clang_minor__ << std::endl;
#elif defined(__GNUC__)
    std::cout << "Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#elif defined(__ICL)
    std::cout << "Compiler: Intel C++ (ICL)" << std::endl;
#elif defined(_MSC_VER)
    std::cout << "Compiler: MSVC " << _MSC_VER << std::endl;
#else
    std::cout << "Compiler: Unknown" << std::endl;
#endif

    // Show platform
    std::cout << "\n=== Platform Detection ===" << std::endl;

#if defined(__linux__)
    std::cout << "Platform: Linux" << std::endl;
#elif defined(_WIN32)
    std::cout << "Platform: Windows" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Platform: macOS" << std::endl;
#else
    std::cout << "Platform: Unknown" << std::endl;
#endif

    // Show C++ standard
    std::cout << "\n=== C++ Standard ===" << std::endl;

#if __cplusplus >= 202002L
    std::cout << "C++ Standard: C++20 or later" << std::endl;
#elif __cplusplus >= 201703L
    std::cout << "C++ Standard: C++17" << std::endl;
#elif __cplusplus >= 201402L
    std::cout << "C++ Standard: C++14" << std::endl;
#elif __cplusplus >= 201103L
    std::cout << "C++ Standard: C++11" << std::endl;
#else
    std::cout << "C++ Standard: Pre-C++11" << std::endl;
#endif

    std::cout << "__cplusplus = " << __cplusplus << std::endl;

    return 0;
}
