// Best practice: 
// 1. If a header file is paired with a code file (e.g. add.h with add.cpp), 
// they should both have the same base name (add).
// 2. Header files should generally not contain function and variable definitions, 
// so as not to violate the one definition rule. An exception is made for symbolic constants

// define a function in a namespace
namespace gongyisheng{
    int sumOfTwoNumbers_gys(int a, int b) {
        return a + b;
    }
}

int sumOfTwoNumbers(int a, int b) {
    return a + b;
};

// function prototype for add.h -- don't forget the semicolon!
int _sumOfTwoNumbers(int a, int b);
