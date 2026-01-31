#include <iostream>

// test: g++ -o build/reference_rvalue basic/reference_rvalue.cpp && build/reference_rvalue

class foo{
    public:
        int a;
        int b;
        foo(){
            this->a = 0;
            this->b = 0;
        }
        foo(int a, int b){
            this->a = a;
            this->b = b;
        }
        // copy, use lvalue reference
        foo(const foo& other){
            this->a = other.a;
            this->b = other.b;
        }
        // move, use rvalue reference
        foo(foo&& other){
            this->a = other.a;
            this->b = other.b;
            other.a = 0;
            other.b = 0;
        }
};

// to be fixed
int main(){
    foo f1 = foo(1,2);
    foo f2 = foo(foo(1,2)); // copy
    foo f3((foo())); // move
    foo f4 = std::move(f3); // move
    foo&& f5 = foo(); // move
    std::cout << f1.a << " " << f1.b << std::endl;
    std::cout << f2.a << " " << f2.b << std::endl;
    return 0;
}