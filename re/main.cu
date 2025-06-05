#include <iostream>
#include "kernels/vectorAdd_cute.cuh"
#include "kernels/hgemm/hgemm_cuda_naive.cuh"
#include "kernels/hgemm/hgemm_cuda_shared_memory.cuh"
#include "kernels/hgemm/hgemm_cuda_thread_coarsening.cuh"
#include "kernels/hgemm/hgemm_cuda_using_half4.cuh"
#include "kernels/hgemm/hgemm_cuda_register_outer_product.cuh"
#include "kernels/hgemm/hgemm_cuda_smem_transpose.cuh"
#include "kernels/hgemm/hgemm_cute_naive.cuh"
#include "kernels/hgemm/hgemm_cute_multiStage.cuh"
#include "cublas_v2.h"

int main(int argc, char **argv)
{
    using T = cute::half_t;
    srand(520);
    constexpr int M = 81920;
    constexpr int N = 256;
    constexpr int K = 256;

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    auto tA = make_tensor(h_A.data(), make_shape(M, K), make_stride(K, 1));
    auto tB = make_tensor(h_B.data(), make_shape(N, K), make_stride(K, 1));

    cpu_rand_data(&tA);
    cpu_rand_data(&tB);

    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    thrust::host_vector<T> h_C_cublas(M * N);
    thrust::device_vector<T> d_C_cublas = h_C_cublas;
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    half alpha = 1.f;
    half beta = 0.f;
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, (half *)d_B.data().get(), N, (half *)d_A.data().get(), K,
                &beta, (half *)d_C_cublas.data().get(), N);
    CUTE_CHECK_LAST();
    h_C_cublas = d_C_cublas;

    // my kernel
    thrust::host_vector<T> h_C_cuda_naive(M * N);
    hgemm_cuda_naive(h_A, h_B, h_C_cuda_naive, M, N, K);
    thrust::host_vector<T> h_C_sharedMemory(M * N);
    hgemm_cuda_sharedMemory(h_A, h_B, h_C_sharedMemory, M, N, K);
    thrust::host_vector<T> h_C_threadCoarsening(M * N);
    hgemm_cuda_threadCoarsening(h_A, h_B, h_C_threadCoarsening, M, N, K);
    thrust::host_vector<T> h_C_usingHalf4(M * N);
    hgemm_cuda_usingHalf4(h_A, h_B, h_C_usingHalf4, M, N, K);
    thrust::host_vector<T> h_C_registerOuter(M * N);
    hgemm_cuda_registerOuter(h_A, h_B, h_C_registerOuter, M, N, K);
    thrust::host_vector<T> h_C_transposeA(M * N);
    hgemm_cuda_transposeA(h_A, h_B, h_C_transposeA, M, N, K);
    thrust::host_vector<T> h_C_cute_naive(M * N);
    hgemm_cute_naive(h_A, h_B, h_C_cute_naive, M, N, K);
    thrust::host_vector<T> h_C_cute_multiStage(M * N);
    hgemm_cute_multiStage(h_A, h_B, h_C_cute_multiStage, M, N, K);

    cpu_compare(h_C_cute_naive,h_C_cute_multiStage);
}