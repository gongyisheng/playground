#include<string>
#include<iostream>

// test: g++ -o build/structure basic/structure.cpp && build/structure

using namespace std;

struct demoStruct{   // Structure declaration, name=demoStruct
  int myNum;         // Member (int variable)
  string myString;   // Member (string variable)
} myStructure, myStructure1, myStructure2;       // Structure variable

demoStruct myStructure4; // Structure variable

int main() {
    myStructure.myNum = 15;    // Accessing members
    myStructure.myString = "Some text";
    cout << myStructure.myNum << endl;
    cout << myStructure.myString << endl;
    return 0;
}