#include <iostream>
using namespace std;

int main() {
  cout << "Hello World! \n"; // c-out, see-out
  cout << "Hello World!" << endl; // use endl to print a new line
  cout << "I am learning C++";
  /*
  endl does two jobs: 
  1. move the cursor to the next line
  2. flush the buffer (print out cached output immediately)
  /n only moves the cursor to the next line
  perfer /n over endl
  */
  return 0; // Your main function should return the value 0 if the program ran normally.
  // The return number is so called "status code" or "exit code".
  // You can also include <cstdlib> and return 0, EXIT_SUCCESS or EXIT_FAILURE to maximize portability
}