#include <mutex>
#include <thread>
#include <iostream>
#include <chrono>

// g++ -std=c++17 -pthread concurrent/mutex.cpp -o build/mutex && ./build/mutex

std::mutex mtx;
int counter = 0;

void increment(int id) {
    for (int i = 0; i < 5; i++) {
        {
            // RAII: locks on construction, auto-unlocks when 'lock' goes out of scope (end of this block)
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
            std::cout << "Thread " << id << ": counter = " << counter << std::endl;
        }  // unlock happens here automatically
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    std::thread t1(increment, 1); // creates and starts t1 immediately
    std::thread t2(increment, 2); // creates and starts t2 immediately
    t1.join(); // main thread waits for t1 to finish
    t2.join(); // main thread waits for t2 to finish
    std::cout << "Final counter: " << counter << std::endl;
    return 0;
}
