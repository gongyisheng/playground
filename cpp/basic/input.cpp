#include <iostream>

// test: g++ -o build/input basic/input.cpp && build/input

using namespace std;

int main() {
    int x; 
    cout << "Type a number: "; // Type a number and press enter
    cin >> x; // Get user input from the keyboard
    cout << "Your number is: " << x << endl; // Display the input value
}