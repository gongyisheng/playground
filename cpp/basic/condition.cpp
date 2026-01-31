#include <string>
#include <iostream>

// test: g++ -o build/condition basic/condition.cpp && build/condition

using namespace std;

int main() {
    int time;
    cout << "What time is it? " << endl;
    cin >> time;

    if (time < 18) {
        cout << "Good day." << endl;
    } else if (time < 24) {
        cout << "Good evening." << endl;
    } else {
        cout << "Invalid." << endl;
    }

    string result = (time < 18) ? "Good day." : "Good evening or invalid.";
    cout << result << endl;
}