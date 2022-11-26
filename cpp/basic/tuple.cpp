#include <tuple>
#include <iostream>
using namespace std;


// Conceptually, Tuples are similar to old data structures (C-like structs)
// but instead of having named data members,
// its elements are accessed by their order in the tuple.

int main() {
    // We start with constructing a tuple.
    // Packing values into tuple
    auto first = make_tuple(10, 'A');
    const int maxN = 1e9;
    const int maxL = 15;
    auto second = make_tuple(maxN, maxL);

    // Printing elements of 'first' tuple
    std::cout << get<0>(first) << " " << get<1>(first) << '\n'; //prints : 10 A

    // Printing elements of 'second' tuple
    cout << get<0>(second) << " " << get<1>(second) << '\n'; // prints: 1000000000 15

    // Unpacking tuple into variables

    int first_int;
    char first_char;
    tie(first_int, first_char) = first;
    cout << first_int << " " << first_char << '\n';  // prints : 10 A

    // We can also create tuple like this.

    tuple<int, char, double> third(11, 'A', 3.14141);
    // tuple_size returns number of elements in a tuple (as a constexpr)

    cout << tuple_size<decltype(third)>::value << '\n'; // prints: 3

    // tuple_cat concatenates the elements of all the tuples in the same order.

    auto concatenated_tuple = tuple_cat(first, second, third);
    // concatenated_tuple becomes = (10, 'A', 1e9, 15, 11, 'A', 3.14141)

    cout << get<0>(concatenated_tuple) << '\n'; // prints: 10
    cout << get<3>(concatenated_tuple) << '\n'; // prints: 15
    cout << get<5>(concatenated_tuple) << '\n'; // prints: 'A'
}