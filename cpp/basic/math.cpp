#include <cmath>
#include <iostream>

// test: g++ -o build/math basic/math.cpp && build/math

using namespace std;

int main() {
    cout << max(5, 10) << endl; // 10
    cout << min(5, 10) << endl; // 5
    cout << sqrt(64) << endl; // 8
    cout << round(2.6) << endl; // 3
    cout << log(2) << endl; // 0.693147
}