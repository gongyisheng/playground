#include <cstdlib>
#include <cstdio>

// test: g++ -o build/memory basic/memory.cpp && build/memory

// WRONG: Single pointer - only modifies local copy
bool allocate_wrong(void* ptr, size_t size) {
    printf("  Inside function: ptr before malloc = %p\n", ptr);
    ptr = malloc(size);
    printf("  Inside function: ptr after malloc  = %p\n", ptr);
    return ptr != nullptr;
}

// CORRECT: Pointer to pointer - modifies the original
bool allocate_correct(void** ptr, size_t size) {
    printf("  Inside function: *ptr before malloc = %p\n", *ptr);
    *ptr = malloc(size);
    printf("  Inside function: *ptr after malloc  = %p\n", *ptr);
    return *ptr != nullptr;
}

int main() {
    // Wrong way
    printf("=== Wrong way (void*) ===\n");
    void* wrong_ptr = nullptr;
    printf("Before call: wrong_ptr = %p\n", wrong_ptr);
    allocate_wrong(wrong_ptr, 100);
    printf("After call:  wrong_ptr = %p\n\n", wrong_ptr);  // Still nullptr!

    // Correct way
    printf("=== Correct way (void**) ===\n");
    void* correct_ptr = nullptr;
    printf("Before call: correct_ptr = %p\n", correct_ptr);
    allocate_correct(&correct_ptr, 100);
    printf("After call:  correct_ptr = %p\n", correct_ptr);  // Has valid address!

    // Cleanup
    free(correct_ptr);
    return 0;
}
