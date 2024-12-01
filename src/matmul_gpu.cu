#include <iostream>
#include <matmul_gpu.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstddef>

#define BLOCK_SIZE 32
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
__global__ void matmul_naive_kernel(T *A, T *B, T* C, size_t m, size_t n, size_t p)
// Treats thread independantly of block
{   
    // for each block, threadIdx resets from 0 -> BLOCK_SIZExBLOCK_SIZE
    // convert thread numbering in block to thread numbering in matrix C
    size_t i = blockIdx.y*blockDim.y + threadIdx.y;
    size_t j = blockIdx.x*blockDim.x + threadIdx.x;

    // Check clipping case if converted thread numbering exceed dim of C
    if (i >= m || j >= p)
    {
        return;
    }

    T sum = 0;
    for (size_t k=0; k<n; k++)
    {
        // Access pos i*n + k of A and pos k*p + j of B in global memory -> Slow
        // Covert i,j to matrix in flatten form
        sum += A[i*n + k] * B[k*p + j];
    }
    C[i*p + j] = sum;
}

template<typename T>
__global__ void matmul_tiled_kernel(T *A, T *B, T* C, size_t m, size_t n, size_t p)
// Used tiled matMul to utilize shared memory of L1 cache amongst all threads in a block
{
    // Shared memory allocation
    // Make it easy to have sub-mat dim = block size, so that thread numbering in block is equal to sub-mat
    __shared__ T subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T subB[BLOCK_SIZE][BLOCK_SIZE];

    T sum=0;

    // t runs left-to-right for A, and top-to-bottom for B
    for (size_t t = 0; t < std::ceil(static_cast<double>(n)/static_cast<double>(BLOCK_SIZE));  t++)
    {   
        // Load memory from matrix in global memory to sub-matrix in shared memory
        // convert thread numbering in block to thread numbering in matrix C        

        // Load A -> subA: same row , different col as C
        size_t i_A = blockIdx.y*blockDim.y + threadIdx.y;
        size_t j_A = t * blockDim.x + threadIdx.x;
        if (j_A >= n)
        {
            subA[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            subA[threadIdx.y][threadIdx.x] = A[i_A * n + j_A];
        }
        

        // Load B -> subB: different row , same col as C
        size_t i_B = blockDim.y * t + threadIdx.y;
        size_t j_B = blockIdx.x*blockDim.x + threadIdx.x;
        if (i_B >= n)
        {
            subB[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            subB[threadIdx.y][threadIdx.x] = B[i_B * p + j_B];
        }
        
        __syncthreads(); // Make sure all threads finished loading shared mem
        // Sub-matrix calculation, used elem in sub-matrix
        for (size_t k=0; k<BLOCK_SIZE; k++)
        {
            sum += subA[threadIdx.y][k] * subB[k][threadIdx.x];
            // Debug: subA[threadIdx.y][k] * subB[k][threadIdx.x]
        }
        __syncthreads();
    }

    // Save accumulated result to C
    size_t i = blockIdx.y*blockDim.y + threadIdx.y;
    size_t j = blockIdx.x*blockDim.x + threadIdx.x;   
    C[i*p + j] = sum; 
}
template<typename T>
void matmul_gpu(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, size_t m, size_t n, size_t p, bool naive)
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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); //(32, 32, 1)
    dim3 gridSize(std::ceil(static_cast<double>(p)/static_cast<double>(BLOCK_SIZE)), 
                std::ceil(static_cast<double>(m)/static_cast<double>(BLOCK_SIZE)));

    if (naive)
    {
        matmul_naive_kernel <<<gridSize, blockSize >>> (d_A, d_B, d_C, m, n, p);
    }
    else
    {
        matmul_tiled_kernel <<<gridSize, blockSize >>> (d_A, d_B, d_C, m, n, p);
    }
    cudaDeviceSynchronize();

    // Copy data from device to host.
    checkCuda(cudaMemcpy(ptrC, d_C, sizeof(T) * C.size(),
                        cudaMemcpyDeviceToHost));    

    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."<< std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Free device buffer.
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));    



}

// Explicit template initialization for type int and type float
template void matmul_gpu<float>(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, size_t, size_t, size_t, bool);
template void matmul_gpu<int>(const std::vector<int>&, const std::vector<int>&, std::vector<int>&, size_t, size_t, size_t, bool);