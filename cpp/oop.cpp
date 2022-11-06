#include <iostream>
#include <string>
using namespace std;

class MyClass {           // The class
    public:               // Access specifier
        int myNum;        // Attribute (int variable)
        string myString;  // Attribute (string variable)
        MyClass(int x) {  // Constructor (can also be defined outside the class)
            cout << "This is Constructor!" << endl;
            myNum = x;
            cout << "x = " << x << endl;
            cout << "myNum = " << myNum << endl;
        }
        void myMethodIn() { // Method/function defined inside the class
            cout << "Hello World!" << endl;
        }
        void myMethodOut(); // Method/function defined outside the class
        
    protected:            // Protected access specifier (can be accessed by derived classes, can't be accessed outside class)
        int protectedNum;       // Attribute (int variable)
        string protectedString; // Attribute (string variable)

    private:                // Private access specifier 
        int privateNum;
        string privateString;// Private attribute
    
    int anotherInt; //by default, all members are private
};

// Method/function definition outside the class
void MyClass::myMethodOut() {
    cout << "Hello World!" << endl;
}

class MyChildClass: public MyClass {
    public:
        MyChildClass(int x): MyClass(x) { // Call the constructor of the base class
            cout << "This is Child Constructor!" << endl;
            cout << "myNum = " << myNum << endl;
            cout << "protectedNum = " << protectedNum << endl;
            // cout << "privateNum = " << privateNum << endl; // error
        }
};

int main() {
    MyClass myObj(1);  // Create an object of MyClass

    // Access attributes and set values
    myObj.myNum = 15; 
    myObj.myString = "Some text";

    // Print attribute values
    cout << myObj.myNum << endl;
    cout << myObj.myString << endl;

    // Call the method
    myObj.myMethodIn();
    myObj.myMethodOut();

    MyChildClass myChildObj(20);
    cout << myChildObj.myNum << endl;
    return 0;
}