cmake_minimum_required(VERSION 3.22)
project(neural_lace_hippocampal_memory_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -ffast-math -Wall -Wextra -pthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

add_library(ca1_stimulation_cuda src/cuda_ca1_stimulation.cu)
set_target_properties(ca1_stimulation_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(neural_lace_engine src/main.cpp)
target_link_libraries(neural_lace_engine PRIVATE ca1_stimulation_cuda CUDA::cudart pthread)

add_executable(neural_lace_test src/test_runner.cpp)
target_link_libraries(neural_lace_test PRIVATE ca1_stimulation_cuda CUDA::cudart pthread)
