#include <iostream>

// test: g++ -o build/cast basic/cast.cpp && build/cast

using namespace std;

int main(){
    // YOU SHOULD KNOW YOU ARE CASTING
    // use static_cast to cast value of variables, low risk
    // static_cast<target_type>(value)
    double score = 96.5;
    int score_ = static_cast<int>(score);
    cout << "static_cast, " << "before cast: " << score << ", after cast: " << score_ << endl;

    // const_cast
    int i = 3;
    const int& rci = i; 
    const_cast<int&>(rci) = 4;
    int j = const_cast<int&>(rci);
    cout << "const_cast, " << "i = " << i << '\n';
    cout << "const_cast, " << "j = " << j << '\n';
    j = 20;
    cout << "const_cast, " << "j = " << j << '\n';

    // dynamic_cast
    // the old and new type should be both pointer or reference
    // cast between Polymorphic
    
    // T1 obj;
    // T2* pObj = dynamic_cast<T2*>(&obj);//转换为T2指针，失败返回NULL
    // T2& refObj = dynamic_cast<T2&>(obj);//转换为T2引用，失败抛出bad_cast异常
}