cmake_minimum_required(VERSION 3.16)
project(intrisic-learning LANGUAGES CXX)

add_executable(bench
    src/main.cpp
    src/core/tensor.cpp
    src/lib/gemm.cpp
    src/lib/gemv.cpp
    src/bench/bench.cpp
)

target_include_directories(bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(bench PRIVATE
    -O3
    -mavx2
    -mfma
    -fopenmp
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
endif()