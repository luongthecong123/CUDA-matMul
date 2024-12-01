#pragma once
#include <vector>
#include <cstddef>
template<typename T>
void matmul_cpu(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, size_t m, size_t n, size_t p)
/*
A: mxn
B: nxp
C: mxp = A@B
*/
{
    for (size_t i=0; i<m; i++) //loop through elem in C, O(n^3)
    {
        for (size_t j=0; j<p; j++)
        {
            T sum = 0;
            for (size_t k=0; k<n; k++)
            {
                sum += A[i*n + k] * B[k*p + j];
            }
            C[i*p + j] = sum;
        }
    }
}