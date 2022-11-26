#include <iostream>
#include <set>
using namespace std;

int main(){
    set<int> ST;    // Will initialize the set of int data type
    ST.insert(30);  // Will insert the value 30 in set ST
    ST.insert(10);  // Will insert the value 10 in set ST
    ST.insert(20);  // Will insert the value 20 in set ST
    ST.insert(30);  // Will insert the value 30 in set ST
    // Now elements of sets are as follows
    //  10 20 30

    // To erase an element
    ST.erase(20);  // Will erase element with value 20
    // Set ST: 10 30
    // To iterate through Set we use iterators
    set<int>::iterator it;
    for(it=ST.begin();it!=ST.end();it++) {
        cout << *it << endl;
    }
    // To clear the complete container we use Container_name.clear()
    ST.clear();
    cout << ST.size();  // will print the size of set ST
    // Output: 0

    // NOTE: for duplicate elements we can use multiset
    // NOTE: For hash sets, use unordered_set. They are more efficient but
    // do not preserve order. unordered_set is available since C++11
}