#include <iostream>
using namespace std;

int main() {
    // initialize a variable
    int a1, a2, a3; // Default initialization
    int a4 = 10;    // Copy initialization
    int a5 = {10};  // Brace initialization    
    int a6(10);     // Direct initialization
 
    /* Unlike some programming languages, 
    C/C++ does not initialize most variables 
    to a given value (such as zero) automatically.
    Thus when a variable is given a memory address 
    to use to store data, the default value of 
    that variable is whatever (garbage) value 
    happens to already be in that memory address! */

    // best practice: always initialize your variables

    // type variableName = value;
    int myInt = 10;
    cout << "myInt is: " << myInt << endl;
    double myDouble = 10.5;
    cout << "myDouble is: " << myDouble << endl;
    char myChar = 'a';
    cout << "myChar is: " << myChar << endl;
    bool myBool = true;
    cout << "myBool is: " << myBool << endl;

    int x = 5, y = 6, z = 50;
    cout << "X+Y+Z = "<< x + y + z << endl;
    return 0;
}