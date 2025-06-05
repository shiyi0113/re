#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

template <unsigned int BLOCK, class T>
__global__ void hgemm_cuda_sharedMemory_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    __shared__ T s_A[BLOCK][BLOCK];
    __shared__ T s_B[BLOCK][BLOCK];

    T *A_begin = d_A + blockIdx.x * blockDim.x * K;
    T *B_begin = d_B + blockIdx.y * blockDim.y;

    T sum;
    sum = 0;
    for (int step = 0; step < (K + BLOCK - 1) / BLOCK; step++)
    {
        s_A[threadIdx.x][threadIdx.y] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCK];
        s_B[threadIdx.x][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCK) * N + threadIdx.y];
        __syncthreads();
        for (int i = 0; i < BLOCK; i++)
            sum += s_A[threadIdx.x][i] * s_B[i][threadIdx.y];
        __syncthreads();
    }

    const int C_m = threadIdx.x + blockIdx.x * blockDim.x;
    const int C_n = threadIdx.y + blockIdx.y * blockDim.y;
    d_C[C_m * N + C_n] = sum;
}
template <class T>
void hgemm_cuda_sharedMemory(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    const int BLOCK = 16;
    dim3 Grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    dim3 Block(BLOCK, BLOCK);
    hgemm_cuda_sharedMemory_kernel<BLOCK, T><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    h_C = d_C;
}