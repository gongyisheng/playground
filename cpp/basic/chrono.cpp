#include <chrono>
#include <iostream>
#include <thread>

// test: g++ -std=c++17 basic/chrono.cpp -o build/chrono && ./build/chrono

int main() {
    using namespace std::chrono;

    // === Durations ===
    std::cout << "=== Durations ===\n";
    seconds sec(2);
    milliseconds ms(1500);
    microseconds us(1000000);

    std::cout << "2 seconds = " << sec.count() << " sec\n";
    // duration_cast<Target>(source): converts duration types, truncates toward zero
    std::cout << "1500 ms = " << duration_cast<seconds>(ms).count() << " sec\n";
    std::cout << "1000000 us = " << duration_cast<milliseconds>(us).count() << " ms\n";

    // duration arithmetic
    auto total = seconds(1) + milliseconds(500);
    std::cout << "1s + 500ms = " << duration_cast<milliseconds>(total).count() << " ms\n\n";

    // === Clocks ===
    std::cout << "=== Clocks ===\n";

    // steady_clock: monotonic, best for measuring intervals
    auto steady_now = steady_clock::now();
    std::cout << "steady_clock is monotonic (never goes backwards)\n";

    // system_clock: wall clock time
    auto sys_now = system_clock::now();
    std::cout << "sys_now (time_since_epoch): " << sys_now.time_since_epoch().count() << "\n";
    // to_time_t: converts time_point to C-style time_t (seconds since Unix epoch)
    auto sys_time = system_clock::to_time_t(sys_now);
    std::cout << "sys_time (time_t): " << sys_time << "\n";
    // ctime: converts time_t to human-readable string (includes newline)
    std::cout << "system_clock (wall time): " << std::ctime(&sys_time);

    // high_resolution_clock: highest precision available
    std::cout << "high_resolution_clock for finest precision\n\n";

    // === Measuring Elapsed Time ===
    std::cout << "=== Measuring Elapsed Time ===\n";
    auto start = steady_clock::now();

    // simulate work
    std::this_thread::sleep_for(milliseconds(150));

    auto end = steady_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start);
    auto elapsed_us = duration_cast<microseconds>(end - start);

    std::cout << "Sleep took: " << elapsed_ms.count() << " ms\n";
    std::cout << "Sleep took: " << elapsed_us.count() << " us\n\n";

    // === Floating Point Durations ===
    std::cout << "=== Floating Point Durations ===\n";
    duration<double> elapsed_sec = end - start;
    duration<double, std::milli> elapsed_ms_float = end - start;
    std::cout << "Elapsed: " << elapsed_sec.count() << " seconds\n";
    std::cout << "Elapsed: " << elapsed_ms_float.count() << " ms\n\n";

    // === Time Point Arithmetic ===
    std::cout << "=== Time Point Arithmetic ===\n";
    auto future = steady_clock::now() + seconds(5);
    auto remaining = duration_cast<seconds>(future - steady_clock::now());
    std::cout << "5 seconds from now, remaining: " << remaining.count() << " sec\n";

    return 0;
}
