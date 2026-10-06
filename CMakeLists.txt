cmake_minimum_required(VERSION 3.18)
project(infernt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(infernt
    src/main.cpp
    src/tensor.cpp
    src/config.cpp
    src/cuda_init.cu
)

target_link_libraries(infernt
    cudart
)
