# Matrix Multiplication: CPU vs CUDA-enabled GPU

### Test Setup
```cpp
/*
A: mxn
B: nxp
C: mxp = A@B
*/
size_t m = 1024;
size_t n = 1024;
size_t p = 1024;
mul<float>(m, n, p);
```

## Performance on Nvidia RTX 3050Ti

- **Time taken for CPU:** 8.37 seconds (8,365,754 microseconds)
- **Time taken for GPU (Naive):** 0.13 seconds (126,666 microseconds)  
- **Time taken for GPU (Tiled):** 0.02 seconds (18,702 microseconds)  

## Build Project

#### Linux:
```bash
mkdir build
cd build
cmake ..
make
```

#### Windows:
```bash
mkdir build
cd build
cmake ..
msbuild ./matmul.sln
```