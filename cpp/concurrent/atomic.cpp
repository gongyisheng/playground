#include <atomic>
#include <thread>
#include <iostream>

// std::atomic<T> API:
//   load()                        - read value atomically
//   store(val)                    - write value atomically
//   exchange(val)                 - swap and return old value
//   compare_exchange_strong(exp, val) - CAS: if current==exp, set to val

// test: g++ -std=c++17 -pthread concurrent/atomic.cpp -o build/atomic && ./build/atomic

std::atomic<uint64_t> atomic_counter{0};
uint64_t regular_counter = 0;

void increment_both() {
    for (int i = 0; i < 100000; i++) {
        atomic_counter++;      // thread-safe, no lock needed
        regular_counter++;     // NOT thread-safe (data race)
    }
}

int main() {
    std::thread t1(increment_both);
    std::thread t2(increment_both);
    t1.join();
    t2.join();

    std::cout << "Atomic counter:  " << atomic_counter.load() << " (always 200000)" << std::endl;
    std::cout << "Regular counter: " << regular_counter << " (likely wrong due to race)" << std::endl;

    // other atomic operations
    std::atomic<int> val{10};
    val.store(20);                              // set value
    int old = val.exchange(30);                 // swap, returns old
    std::cout << "\nExchange: old=" << old << ", new=" << val.load() << std::endl;  // load() reads value atomically

    int expected = 30;
    val.compare_exchange_strong(expected, 40);  // CAS operation
    std::cout << "After CAS: " << val.load() << std::endl;

    return 0;
}
