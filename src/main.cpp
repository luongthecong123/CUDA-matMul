#include "matmul_cpu.h"
#include "matmul_gpu.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>

template<typename T>
void mul(size_t m, size_t n, size_t p)
/*
A: mxn, [[row_1], [row_2], ...[row_n]]
B: nxp  [[col_1], [col_2], ...[col_n]]
C: mxp = A@B
*/
{
    // Can use auto intead of std::vector<T>
    std::vector<T> const A{create_rand_vector<T>(m*n)}; // same as std::vector<int> vec1{1, 2, 3}; more memory efficient
    std::vector<T> const B{create_rand_vector<T>(n*p)};
    std::vector<T> C(m*p);


    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu<T>(A, B, C, m, n, p);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken for CPU: " << duration.count() << " microseconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    matmul_gpu<T>(A, B, C, m, n, p);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken for GPU: " << duration.count() << " microseconds" << std::endl;

}

int main()
{
    size_t m = 1024;
    size_t n = 1024;
    size_t p = 1024;

    // A: mxn
    // B: nxp
    // C: mxp
    mul<int>(m, n, p);

    return 0;
}