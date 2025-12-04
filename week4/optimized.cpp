```cpp
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, long long param1, long long param2) {
    double result = 1.0;
    
    for (long long i = 1; i <= iterations; ++i) {
        long long j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double result = calculate(200'000'000, 4, 1) * 4;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << diff.count() << " seconds" << std::endl;
    
    return 0;
}
```