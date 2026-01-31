#include <thread>
#include <iostream>
#include <chrono>

// test: g++ -std=c++17 -pthread concurrent/thread.cpp -o build/thread && ./build/thread

void worker(int id) {
    // get_id: get current thread's unique ID
    std::cout << "Thread " << id << " started (ID: " << std::this_thread::get_id() << ")\n";

    // sleep_for: pause for a duration
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Thread " << id << " woke up after 100ms\n";

    // yield: hint scheduler to let other threads run
    std::this_thread::yield();
    std::cout << "Thread " << id << " yielded and resumed\n";

    // sleep_until: pause until a specific time point
    auto wake_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    std::this_thread::sleep_until(wake_time);
    std::cout << "Thread " << id << " finished\n";
}

void detached_worker() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "Detached thread finished (may not print if main exits first)\n";
}

int main() {
    std::cout << "=== Thread Info ===\n";
    std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n";
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " cores\n\n";

    // joinable: true when thread represents an active execution
    // - true: after construction with a function (running or finished but not joined)
    // - false: default constructed, after join(), after detach(), after move()
    // IMPORTANT: must call join() or detach() before destruction, otherwise std::terminate()
    std::thread t1(worker, 1);
    std::cout << "t1 joinable after creation: " << std::boolalpha << t1.joinable() << "\n";
    std::cout << "t1 ID: " << t1.get_id() << "\n";
    std::cout << "t1 native_handle: " << t1.native_handle() << "\n\n";  // OS-level handle (pthread_t on Linux)

    std::thread t2(worker, 2);

    t1.join();
    std::cout << "\nt1 joinable after join: " << t1.joinable() << "\n";

    t2.join();

    // detach: fire-and-forget, thread runs independently
    // use for: logging, analytics, background tasks where you don't need the result
    // avoid when: you need the result, thread accesses local/shared resources, or must complete before exit
    std::cout << "\n=== Detach Demo ===\n";
    std::thread t3(detached_worker);
    std::cout << "t3 joinable before detach: " << t3.joinable() << "\n";
    t3.detach();
    std::cout << "t3 joinable after detach: " << t3.joinable() << "\n";

    // give detached thread time to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "\nAll done\n";
    return 0;
}
