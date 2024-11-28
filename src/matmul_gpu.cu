#include <iostream>
#include <matmul_gpu.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template<typename T>
__global__ void matmul_gpu_kernel(T *A, T *B, T* C, size_t m, size_t n, size_t p)
{
    size_t i = blockIdx.y*blockDim.y + threadIdx.y;
    size_t j = blockIdx.x*blockDim.x + threadIdx.x;

    // Check clipping case
    if (i >= m || j >= p)
    {
        return;
    }

    T sum = 0;
    for (size_t k=0; k<n; k++)
    {
        sum += A[i*n + k] * B[k*p + j];
    }
    C[i*p + j] = sum;
}


template<typename T>
void matmul_gpu(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, size_t m, size_t n, size_t p)
{

    // Ptr to vectors, to convert it to C syntax
    T const* ptrA = A.data();
    T const* ptrB = B.data();
    T* ptrC = C.data();

    // Device ptr
    T *d_A, *d_B, *d_C;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_A, sizeof(T) * A.size()));
    checkCuda(cudaMalloc(&d_B, sizeof(T) * B.size()));
    checkCuda(cudaMalloc(&d_C, sizeof(T) * C.size()));
    
    // Copy data from host to device
    checkCuda(cudaMemcpy(d_A, ptrA, sizeof(T) * A.size(), cudaMemcpyHostToDevice));    
    checkCuda(cudaMemcpy(d_B, ptrB, sizeof(T) * B.size(), cudaMemcpyHostToDevice));  

    size_t blockDim = 32;
    dim3 blockSize(blockDim, blockDim); //(32, 32, 1)
    dim3 gridSize(std::ceil(static_cast<double>(p)/static_cast<double>(blockDim)), 
                std::ceil(static_cast<double>(m)/static_cast<double>(blockDim)));

    matmul_gpu_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, p);

    cudaDeviceSynchronize();

    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Free device buffer.
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));    



}

// Explicit template initialization for type int and type float
template void matmul_gpu<float>(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, size_t, size_t, size_t);
template void matmul_gpu<int>(const std::vector<int>&, const std::vector<int>&, std::vector<int>&, size_t, size_t, size_t);