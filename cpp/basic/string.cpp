#include <string>
#include <iostream>
using namespace std;

int main() {
    string myString1 = "Hello";
    cout << myString1 << endl;
    // concatenate - add
    string firstName1 = "John ";
    string lastName1 = "Doe";
    string fullName1 = firstName1 + lastName1;
    cout << fullName1 << endl;
    // string z = "10" + 20; RAISE ERROR!

    // concatenate - append
    string firstName2 = "John ";
    string lastName2 = "Doe";
    string fullName2 = firstName2.append(lastName2);
    cout << fullName2 << endl;

    // length
    string txt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    cout << "The length of the txt string is: " << txt.length() << endl;
    cout << "The length of the txt string is: " << txt.size() << endl; // size() is alias of length()

    // access characters - lookup
    string myString2 = "Hello";
    cout << myString2[0] << " " << myString2[1] << endl;

    // access characters - update
    myString2[0] = 'J';
    cout << myString2 << endl;

    // special characters
    string txt2 = "It\'s alright.";
    cout << txt2 << endl;
    string txt3 = "The character \\ is called backslash.";
    cout << txt3 << endl;

    return 0;
}