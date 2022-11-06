#include<iostream>
#include<string>
using namespace std;

int main() {
    // pointer <--> address <--> variable
    // reference <--> variable
    // &: reference/get address
    // *: dereference/get value
    string food = "Pizza";
    string &meal = food; // & is a reference operator, meal is a reference to food
    string* foodptr = &food; // foodptr is a pointer to food
    string anotherMeal = food; // anotherMeal is a copy of food

    cout << food << endl;  // Outputs Pizza
    cout << &food << endl; // Get memory address of food.
    cout << meal << endl;  // Outputs Pizza
    cout << &meal << endl; // Get memory address of meal.
    cout << anotherMeal << endl;  // Outputs Pizza
    cout << &anotherMeal << endl; // Get memory address of anotherMeal.
    cout << foodptr << endl;  // Outputs `food` address
    cout << *foodptr << endl; // Outputs Pizza, dereference pointer

    *foodptr = "Hamburger"; // Change value of food through pointer
    cout << foodptr << endl;  // Outputs new `food` address
    cout << *foodptr << endl; // Outputs Hamburger
    return 0;
}