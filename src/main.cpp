#include "matmul_cpu.h"
#include "matmul_gpu.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstddef>

template<typename T>
void mul(size_t m, size_t n, size_t p)
/*
A: mxn, [[row_1], [row_2], ...[row_n]]
B: nxp  [[col_1], [col_2], ...[col_n]]
C: mxp = A@B
*/
{
    // Can use auto intead of std::vector<T>
    std::vector<T> const A{ create_rand_vector<T>(m * n) }; // same as std::vector<int> vec1{1, 2, 3}; more memory efficient
    std::vector<T> const B{ create_rand_vector<T>(n * p) };

    //std::cout << "A: " << "\n";
    //printVec(A, m, n);
    //std::cout << "B: " << "\n";
    //printVec(B, n, p);

    // std::vector<T> const A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // std::vector<T> const B{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

    std::vector<T> C(m * p); // CPU
    std::vector<T> D(m * p); // GPU naive
    std::vector<T> E(m * p); // GPU tiled


    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu<T>(A, B, C, m, n, p);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


    //std::cout << "C: " << "\n";
    //printVec(C, m, p);
    std::cout << "Time taken for CPU: " << duration.count() << " microseconds" << std::endl;
    ///////////////////////////////////// Naive //////////////////////////////////////////////
    start = std::chrono::high_resolution_clock::now();
    matmul_gpu<T>(A, B, D, m, n, p, true);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    //std::cout << "D: " << "\n";
    //printVec(D, m, p);
    std::cout << "Time taken for GPU naive: " << duration.count() << " microseconds" << std::endl;

    // printVec(A, m, n);
    // printVec(B, n, p);
    // printVec(D, m, p);


    double tolerance = 1e-4;
    if (allclose<T>(C, D, tolerance))
    {
        std::cout << "CPU and GPU matmult naive are close, error within: " << tolerance << "\n";
    }
    else { std::cout << "CPU and GPU matmult naive are not close\n"; }

    ///////////////////////////////////// Tiled //////////////////////////////////////////////
    start = std::chrono::high_resolution_clock::now();
    matmul_gpu<T>(A, B, E, m, n, p, false);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    //std::cout << "D: " << "\n";
    //printVec(D, m, p);
    std::cout << "Time taken for GPU tiled: " << duration.count() << " microseconds" << std::endl;

    if (allclose<T>(C, E, tolerance))
    {
        std::cout << "CPU and GPU matmult tiled are close, error within: " << tolerance << "\n";
    }
    else { std::cout << "CPU and GPU matmult tiled are not close\n"; }
}

int main()
{
    size_t m = 1024;
    size_t n = 1024;
    size_t p = 1024;

    // A: mxn
    // B: nxp
    // C: mxp
    mul<float>(m, n, p);

    return 0;
}