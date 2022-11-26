// need to include both iostream and fstream if you want to operate on files
#include <iostream>
#include <fstream> // file stream, a combination of ofstream and ifstream

using namespace std;

int main() {
  // Create and open a text file
  ofstream MyFile("filename.txt");

  // Write to the file
  MyFile << "Files can be tricky, but it is fun enough!";

  // Close the file
  MyFile.close();
}