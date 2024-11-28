# Matrix Multiplication: CPU vs CUDA-enabled GPU

## Performance on Nvidia RTX 30 Series

- **Time taken for CPU:** 7.92 seconds (7918241 microseconds)
- **Time taken for GPU:** 0.27 seconds (271797 microseconds)

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