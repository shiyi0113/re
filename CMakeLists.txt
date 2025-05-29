cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 89)

project(cuda_practice CUDA CXX C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
find_package(CUDAToolkit REQUIRED)
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/third_party/cutlass CACHE PATH "Path to CUTLASS repository")
set(CUTLASS_TOOL_DIR ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools CACHE PATH "Path to CUTLASS_TOOL repository")

include_directories(
        ${PROJECT_SOURCE_DIR}/re
        ${CUTLASS_DIR}/include
        ${CUTLASS_TOOL_DIR}/include
)

file(GLOB_RECURSE SOURCES
        "re/*.cu"
)
file(GLOB_RECURSE CUH_HEADERS
        "re/core/*cuh"
)

add_executable(cute_gemm ${SOURCES})
target_link_libraries(cute_gemm
        PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        #CUDA::cufft
        #CUDA::curand
)

set_target_properties(cute_gemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
