#include <iostream>
using namespace std;

int main() {
    int x = 6;
    cout << "x is: " << x << endl;
    x += 1; // add
    cout << "x is: " << x << " after x += 1" << endl;
    x -= 1; // subtract
    cout << "x is: " << x << " after x -= 1" << endl;
    x *= 2; // multiply
    cout << "x is: " << x << " after x *= 2" << endl;
    x /= 2; // divide
    cout << "x is: " << x << " after x /= 2" << endl;
    x %= 2; // modulus
    cout << "x is: " << x << " after x %= 2" << endl;
    x &= 3; // bitwise AND
    cout << "x is: " << x << " after x &= 3" << endl;
    x |= 3; // bitwise OR
    cout << "x is: " << x << " after x |= 3" << endl;
    x ^= 2; // bitwise XOR
    cout << "x is: " << x << " after x ^= 2" << endl;
    x <<= 3;// bitwise left shift
    cout << "x is: " << x << " after x <<= 3" << endl;
    x >>= 3;// bitwise right shift
    cout << "x is: " << x << " after x >>= 3" << endl;

    // && logical AND
    // || logical OR
    // ! logical NOT
    
    return 0;
}