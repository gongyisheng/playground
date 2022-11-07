#include <iostream>
#include "header_sum.h"
using namespace std;

int _sumOfTwoNumbers(int a, int b) {
    return a + b;
}

int main() {
    int a = 5;
    int b = 10;
    cout << sumOfTwoNumbers(a, b) << endl;
    cout << _sumOfTwoNumbers(a, b) << endl;
    cout << gongyisheng::sumOfTwoNumbers_gys(a, b) << endl; // use function in another namespace
    return 0;
}