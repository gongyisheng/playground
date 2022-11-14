#include<map>
#include<iostream>
using namespace std;

int main(){
    map<char, int> mymap;  // Will initialize the map with key as char and value as int

    mymap.insert(pair<char,int>('A',1));
    // Will insert value 1 for key A
    mymap.insert(pair<char,int>('Z',26));
    // Will insert value 26 for key Z

    // To iterate
    map<char,int>::iterator it;
    for (it=mymap.begin(); it!=mymap.end(); ++it)
        std::cout << it->first << "->" << it->second << std::endl;
    // Output:
    // A->1
    // Z->26

    // To find the value corresponding to a key
    it = mymap.find('Z');
    cout << it->second;
}

// Output: 26

// NOTE: For hash maps, use unordered_map. They are more efficient but do
// not preserve order. unordered_map is available since C++11.