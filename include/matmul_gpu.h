#pragma once
#include <vector>

template<typename T>
void matmul_gpu(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, size_t m, size_t n, size_t p, bool naive);

// Explicit template initialization declaration for type int and type float
extern template void matmul_gpu<float>(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, size_t, size_t, size_t, bool);
extern template void matmul_gpu<int>(const std::vector<int>&, const std::vector<int>&, std::vector<int>&, size_t, size_t, size_t, bool);