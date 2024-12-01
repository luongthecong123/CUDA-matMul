# Matrix Multiplication: CPU vs CUDA-enabled GPU

## Performance on Nvidia RTX 30 Series

- **Time taken for CPU:** 7.92 seconds (7,684,532 microseconds)
- **Time taken for GPU (Naive):** 0.16 seconds (162,475 microseconds)  
  - **CPU and GPU (Naive):** Results are close, with an error within `0.0001`.
- **Time taken for GPU (Tiled):** 0.019 seconds (19,206 microseconds)  
  - **CPU and GPU (Tiled):** Results are close, with an error within `0.0001`.

## Build Project

### First, delete the `build` folder.

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