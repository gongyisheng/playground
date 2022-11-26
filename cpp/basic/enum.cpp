#include <iostream>
using namespace std;

enum ECarTypes : uint8_t
{
  Sedan,     // 0
  Hatchback, // 1
  SUV = 254, // 254
  Wagon      // 255
};

ECarTypes GetPreferredCarType()
{
    return ECarTypes::Hatchback;
}

int main()
{
    ECarTypes preferredCarType = GetPreferredCarType();
    cout << "Preferred car type: " << preferredCarType << endl;
    return 0;
}