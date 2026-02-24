#include <iostream>
#include <vector>
#include <string>

// test: g++ -o build/vector basic/vector.cpp && build/vector

using namespace std;

int main() {
    // Vector (Dynamic array)
    // Allow us to Define the Array or list of objects at run time
    string val;
    vector<string> my_vector; // initialize the vector
    vector<int> my_vector2(5); // initialize the vector with 5 elements
    vector<int> my_vector3(5, 10); // initialize the vector with 5 elements and each element is 10
    vector<int> v = {1,2,3,4}; // initialize the vector with numbers
    cin >> val;
    
    my_vector.push_back(val); // will push the value of 'val' into vector ("array") my_vector
    my_vector.push_back(val); // will push the value into the vector again (now having two elements)

    // To iterate through a vector we have 2 choices:
    // Either classic looping (iterating through the vector from index 0 to its last index):
    for (int i = 0; i < my_vector.size(); i++) {
        cout << my_vector[i] << endl; // for accessing a vector's element we can use the operator []
    }

    // or using an iterator:
    vector<string>::iterator it; // initialize the iterator for vector
    for (it = my_vector.begin(); it != my_vector.end(); ++it) {
        cout << *it << endl;
    }
}
