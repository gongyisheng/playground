#include <iostream>
using namespace std;

// operators can be overloaded:
// +, -, *, /, %, ^, &, |, ~, !, =, <, >, +=, -=, *=, /=, %=, ^=, &=, |=, <<=, >>=, ==, !=, <=, >=, &&, ||, ++, --, <<, >>, [], (), ->, new, delete, new[], delete[]
// operators cannot be overloaded:
// ::, .*, ., ?:, sizeof, typeid, const_cast, dynamic_cast, reinterpret_cast, static_cast
 
class Box
{
   public:
 
      double getVolume(void)
      {
         return length * breadth * height;
      }
      void setLength( double len )
      {
          length = len;
      }
 
      void setBreadth( double bre )
      {
          breadth = bre;
      }
 
      void setHeight( double hei )
      {
          height = hei;
      }
      // operator overload +
      Box operator+(const Box& b)
      {
         Box box;
         box.length = this->length + b.length;
         box.breadth = this->breadth + b.breadth;
         box.height = this->height + b.height;
         return box;
      }
      // operator overload ()
      Box operator()(int a, int b, int c)
      {
         Box box;
         box.length = a*this->length;
         box.breadth = b*this->breadth;
         box.height = c*this->height;
         return box;
      }
      // operator overload =
      // make it not copyable
      // Box operator=(Box&) = delete;
      // make it not movable
      // Box operator=(Box&&) = delete;
   private:
      double length;
      double breadth;
      double height;
};

int main()
{
   Box Box1;
   Box Box2;
   Box Box3;    
   Box Box4; 
   double volume = 0.0;
 
   // Box1
   Box1.setLength(6.0); 
   Box1.setBreadth(7.0); 
   Box1.setHeight(5.0);
 
   // Box2
   Box2.setLength(12.0); 
   Box2.setBreadth(13.0); 
   Box2.setHeight(10.0);
 
   // Box1
   volume = Box1.getVolume();
   cout << "Volume of Box1 : " << volume <<endl;
 
   // Box2
   volume = Box2.getVolume();
   cout << "Volume of Box2 : " << volume <<endl;
 
   // Box3 = Box1+Box2
   Box3 = Box1 + Box2;
 
   // Box3
   volume = Box3.getVolume();
   cout << "Volume of Box3 : " << volume <<endl;

   // Box4
   Box4 = Box1(2, 3, 4);
   volume = Box4.getVolume();
   cout << "Volume of Box4 : " << volume <<endl;
 
   return 0;
}