#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <iomanip>

class LCG {
private:
    uint64_t value;
    static constexpr uint64_t A = 1664525;
    static constexpr uint64_t C = 1013904223;
    static constexpr uint64_t M = 1ULL << 32;

public:
    LCG(uint64_t seed) : value(seed) {}

    uint64_t next() {
        value = (A * value + C) % M;
        return value;
    }
};

int64_t max_subarray_sum(int n, uint64_t seed, int min_val, int max_val) {
    LCG lcg(seed);
    std::vector<int> random_numbers(n);

    // Generar números aleatorios
    for (int i = 0; i < n; ++i) {
        random_numbers[i] = (lcg.next() % (max_val - min_val + 1)) + min_val;
    }

    int64_t max_sum = std::numeric_limits<int64_t>::min();

    // Algoritmo de suma máxima de subarray O(n^2)
    for (int i = 0; i < n; ++i) {
        int64_t current_sum = 0;
        for (int j = i; j < n; ++j) {
            current_sum += random_numbers[j];
            max_sum = std::max(max_sum, current_sum);
        }
    }

    return max_sum;
}

int64_t total_max_subarray_sum(int n, uint64_t initial_seed, int min_val, int max_val) {
    LCG lcg(initial_seed);
    int64_t total_sum = 0;

    for (int i = 0; i < 20; ++i) {
        uint64_t seed = lcg.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }

    return total_sum;
}

int main() {
    const int n = 10000;
    const uint64_t initial_seed = 42;
    const int min_val = -10;
    const int max_val = 10;

    std::cout << "Iniciando cálculo para N=" << n << " y 20 corridas..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << std::string(40, '-') << std::endl;
    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    return 0;
}