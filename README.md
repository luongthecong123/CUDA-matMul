Matrix multiplication: CPU  vs CUDA-supported GPU
On Nvidia RTX 30 series:
Time taken for CPU: 7918241 microseconds (7.92s)
Time taken for GPU: 271797 microseconds  (0.27s)

-- Build project --
First delete "build" folder

Linux:

mkdir build
cd build
cmake ..
make

Windows:

mkdir build
cd build
cmake ..
msbuild ./matmul.sln
