cmake_minimum_required(VERSION 3.25)

project(matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

set(SOURCES 
    src/main.cpp
    src/matmul_gpu.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} CUDA::cudart)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
