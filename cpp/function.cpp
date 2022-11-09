#include <iostream>
using namespace std;

// Best Practice:
// Make sure your functions with non-void return types return a value in all cases.
// Failure to return a value from a value-returning function will cause undefined behavior.

void myFunction() {
    cout << "I just got executed!" << endl;
}

void myFunctionWithParam(int x=1) {
    cout << "I just got executed! x=" << x << endl;
}

void myFunctionArray(int myNumbers[5]) {
    for (int i = 0; i < 5; i++) {
        cout << myNumbers[i] << " ";
    }
    cout << endl;
}

void swapNums(int &x, int &y) {
    // x and y are references
    int z = x;
    x = y;
    y = z;
}

// This function doesn't work
void swapNums2(int x, int y) {
    // x and y are new copies of the original variables
    int z = x;
    x = y;
    y = z;
}

// function overloading
int plusFuncInt(int x, int y) {
  return x + y;
}

double plusFuncDouble(double x, double y) {
  return x + y;
}

void earlyReturn() {
    int x = 20;
    return; // early return
    int y = 30; // unreachable code
}

int main() {
    myFunction();
    myFunctionWithParam(); // default value
    myFunctionWithParam(5); // custom value
    int a = 10, b = 20;
    cout << "a=" << a << ", b=" << b << endl;
    swapNums(a, b);
    cout << "a=" << a << ", b=" << b << endl;
    swapNums2(a, b);
    cout << "a=" << a << ", b=" << b << endl;

    int myNum1 = plusFuncInt(8, 5);
    double myNum2 = plusFuncDouble(4.3, 6.26);
    cout << "Int: " << myNum1 << endl;
    cout << "Double: " << myNum2 << endl;
    return 0;
}