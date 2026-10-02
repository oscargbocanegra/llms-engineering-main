#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <cstdlib>

using namespace std;

class LCG {
public:
    LCG(unsigned long long seed, unsigned long long a = 1664525, unsigned long long c = 1013904223, unsigned long long m = 1ULL << 32)
        : current(seed), a(a), c(c), m(m) {}

    unsigned long long next() {
        current = (a * current + c) % m;
        return current;
    }

private:
    unsigned long long current, a, c, m;
};

// Función para calcular la Suma Máxima del Subarray
long long max_subarray_sum(int n, unsigned long long seed, int min_val, int max_val) {
    LCG lcg(seed);
    vector<int> random_numbers(n);
    
    // Genera 'n' números pseudoaleatorios
    for (int i = 0; i < n; ++i) {
        random_numbers[i] = (lcg.next() % (max_val - min_val + 1)) + min_val;
    }

    long long max_sum = numeric_limits<long long>::min(); // Inicializa con menos infinito

    // Bucle de fuerza bruta O(n^2)
    for (int i = 0; i < n; ++i) {
        long long current_sum = 0;
        for (int j = i; j < n; ++j) {
            current_sum += random_numbers[j];
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }

    return max_sum;
}

// Función para ejecutar 'max_subarray_sum' 20 veces
long long total_max_subarray_sum(int n, unsigned long long initial_seed, int min_val, int max_val) {
    long long total_sum = 0;
    LCG lcg(initial_seed);

    for (int i = 0; i < 20; ++i) {
        unsigned long long seed = lcg.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }

    return total_sum;
}

int main() {
    // --- Parámetros ---
    int n = 10000;            // Número de random numbers (longitud del array)
    unsigned long long initial_seed = 42;   // Initial seed para el LCG principal
    int min_val = -10;        // Minimum value of random numbers
    int max_val = 10;         // Maximum value of random numbers

    cout << "Iniciando cálculo para N=" << n << " y 20 corridas..." << endl;

    auto start_time = chrono::high_resolution_clock::now();

    // Llama a la función principal que realiza las 20 corridas
    long long result = total_max_subarray_sum(n, initial_seed, min_val, max_val);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> execution_time = end_time - start_time;

    // --- Resultados ---
    cout << string(40, '-') << endl;
    cout << "Total Maximum Subarray Sum (20 runs): " << result << endl;
    cout << "Execution Time: " << execution_time.count() << " seconds" << endl;
    cout << string(40, '-') << endl;

    return 0;
}