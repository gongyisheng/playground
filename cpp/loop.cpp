#include<iostream>
using namespace std;

int main() {
    int i = 0;
    // while
    while (i < 5) {
        cout << i << " ";
        i++;
    }
    cout << endl;

    // for
    for(int i = 0; i < 5; i++) {
        cout << i << " ";
    }
    cout << endl;

    // for each
    int myNumbers[5] = {0, 1, 2, 3, 4};
    for (int i : myNumbers) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}