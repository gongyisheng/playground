#include <iostream>
using namespace std;

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

// FIXME: not working 
// Need a better understanding of references, pointers and variable
void swapNums2(int &x, int &y) {
    // x and y are references
    int* z = &x;
    x = *(&y);
    y = *z;
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
    return 0;
}