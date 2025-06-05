#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <class T>
__global__ void hgemm_cuda_naive_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    T *A_begin = d_A + blockDim.x * blockIdx.x * K;
    T *B_begin = d_B + blockDim.y * blockIdx.y;
    T sum;
    sum = 0;
    for (int k = 0; k < K; k++)
    {
        sum += A_begin[threadIdx.x * K + k] * B_begin[k * N + threadIdx.y];
    }
    const int C_m = blockDim.x * blockIdx.x + threadIdx.x;
    const int C_n = blockDim.y * blockIdx.y + threadIdx.y;
    d_C[C_m * N + C_n] = sum;
}
template <class T>
void hgemm_cuda_naive(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;

    const int BLOCK = 16;
    dim3 Grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    dim3 Block(BLOCK, BLOCK);
    hgemm_cuda_naive_kernel<<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    CUTE_CHECK_LAST();
    h_C = d_C;
}
