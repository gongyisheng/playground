// The #define directive can be used to create a macro. 
// In C++, a macro is a rule that defines how input text is converted into replacement output text.
// Macros only cause text substitution for normal code.
// There are two basic types of macros: object-like macros, and function-like macros.

#include <iostream>
// object-like macros
#define E             // any further occurrence of the identifier is removed and replaced by nothing!
#define PI 3.14159265 // replace PI with 3.14159265
#define NEWLINE '\n'  // replace NEWLINE with '\n'

int main() {
    std::cout << "Value of PI: " << PI << std::endl;
    // The preprocessor converts the macros into following:
    // std::cout << "Value of PI: " << 3.14159265 << std::endl;
    return 0;
}

// example fo indef:
#include <iostream>
#define PRINT_JOE
int main(){
    // print out
    #ifdef PRINT_JOE
        std::cout << "Joe\n"; // will be compiled since PRINT_JOE is defined
    #endif

    // ignored
    #ifdef PRINT_BOB
        std::cout << "Bob\n"; // will be ignored since PRINT_BOB is not defined
    #endif
    return 0;
}

// example of if 0:
#include <iostream>
int main(){
    std::cout << "Joe\n";

    #if 0 // Don't compile anything starting here
        std::cout << "Bob\n";
        std::cout << "Steve\n";
    #endif // until this point

    return 0;
}

// IMPORTANT: The #define directive will work beyond scope of the file it is defined in.
// Even if you write #define inside a function, or in another file, it will still work.