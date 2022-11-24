#include<iostream>
#include<string>
using namespace std;
typedef void (*f_ptr)(); 

int func1(){
    return 0;
}

void aaa(){
    cout << "aaa" << endl ;
}

f_ptr f(){ // return a function pointer
    return aaa;
}

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

    // function pointer
    int (*funcptr)() = func1;
    // call function through pointer
    funcptr();

    f_ptr (*ff)() = f; // ff is a function pointer, return a function pointer
    ff()(); // call ff()

    //() > [] > anything else
    int arr[128] = {0};  // arr is an array with int
    int (*arrptr)[128] = &arr; // arrptr is a pointer to array with int
    int *ptr = arr; // ptr is a pointer to array, but it doesn't know it's an array
    cout << arr << endl; // Get memory address of arr.
    cout << &arr << endl; // Get memory address of arr.
    cout << arrptr << endl; // Get memory address of arrptr.
    cout << ptr << endl; // Get memory address of ptr.

    return 0;
}