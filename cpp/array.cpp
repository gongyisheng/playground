#include<iostream>
#include<string>
using namespace std;

int main() {
    string cars1[4];
    string cars2[4] = {"Volvo", "BMW", "Ford", "Mazda"};
    int myNum[3] = {10, 20, 30};
    int myNum2[] = {10, 20, 30}; // omit array size
    int myNum3[3] = {10, 20}; // the rest will be 0

    // access - lookup
    cout << cars2[0] << endl;
    // access - update
    cars2[0] = "Opel";
    cout << cars2[0] << endl;
    // size
    cout << "myNum size: " << sizeof(myNum) << endl;
    // length
    cout << "myNum length: " << sizeof(myNum) / sizeof(int) << endl;

    // multi-dimensional array
    int myNumbers[2][3] = {{1, 2, 3}, {4, 5, 6}};
}