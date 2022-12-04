#include <iostream>
#include <thread>

void myprint()
{
  std::cout << "thread started" << std::endl;
  // ...

  std::cout << "thread stoped" << std::endl;
}

int main()
{
    // create thread
    std::thread t1(myprint); // myprint is callable object 

    // block main thread, main thread will wait until t1 thread finish
    t1.join(); // after join it becomes not joinable

    // detach thread (the thread will run in background, no attachment to main thread)
    // t1.detach();

    std::thread t2(myprint);
    // check if thread is joinable
    if (t2.joinable()) {
        std::cout << "before detach:joinable = true" << std::endl;
    }
    else {
        std::cout << "before detach:joinable = false" << std::endl;
    }
    t2.detach();
    if (t2.joinable()) {
        std::cout << "after detach:joinable = true" << std::endl;
    }
    else {
        std::cout << "after detach:joinable = false" << std::endl;
    }

    std::cout << "in main" << std::endl;

    return 0;
}