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

    // conversion(char*, char[], string)
    // char[] to char*
    char ch[] = "Hello";
    char *p = ch;
    cout << "char[] to char*: " << p << endl; // print out Hello

    // char[] to string
    char ch4[10] = "Hello";
    string s;
    s = ch4;
    cout << "char[] to string: " << s << endl; 

    // char* to char[]
    // 1. use strncpy
    char *p2 = (char *)"Hello";
    char ch2[6];
    strncpy(ch2, p2, 6); // strncpy(destination, source, size), which is more safe than strcpy(unbounded)
    cout << "char* to char[], 1. use strncpy: " << ch2 << endl; // print out Hello
    // 2. use loop
    char ch3[100];
    char* p3 = (char *)"abcdef";
    int i = 0;
    while (*p != '\0') {
        ch3[i++] = *p++;
    }
    ch3[i] = '\0'; // need to add '\0' at the end
    cout << "char* to char[], 2. use loop: " << ch3 << endl;

    // char* to string
    // 1. direct
    string s5;
    char *ch8 = (char *)"Hello";
    s5 = ch8;
    cout << "char* to string, 1. direct: " << s5 << endl;
    // 2. use assign
    string s6;
    char* ch9 = (char *)"abcdef";
    s6 = ch9;
    s6.assign(ch9);
    cout << "char* to string, 2. use assign: " << s6 << endl;

    // string to char*
    // 1. c_str()
    string s2 = "Hello";
    char *ch5;
    ch5 = (char *)s2.c_str(); // c_str() returns a const char* pointing to an array that contains a null-terminated sequence of characters (i.e., a C-string) representing the current value of the string object.
    cout << "string to char*, 1. c_str: " << ch5 << endl;
    // 2. data()
    string s3 = "Hello";
    char *ch6;
    s3.append("\0"); // need to add '\0' at the end
    ch6 = (char *)s3.data(); // data() returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string) representing the current value of the string object.
    cout << "string to char*, 2. data: " << ch6 << endl;
    // 3. copy()
    string s4 = "Hello";
    s4.append("\0"); // need to add '\0' at the end
    char *ch7 = new char[s4.length()];
    s4.copy(ch7, s4.length(), 0); // copy(destination, size, position)
    cout << "string to char*, 3. copy: " << ch7 << endl;

    return 0;
}