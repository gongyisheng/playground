#include <iostream>
#include <fstream>
#include <limits>

// test: g++ -o build/avoid_crash basic/avoid_crash.cpp && build/avoid_crash

using namespace std;

int main(){
    int a = 0;
    cin.clear(); // reset any error flags
    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore any characters in the input buffer until we find an enter character
    cin.get(); // get one more char from the user
    return 0;
}