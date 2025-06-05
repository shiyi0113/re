#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

template <class T, unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void hgemm_cuda_threadCoarsening_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    __shared__ T s_A[BLOCKNUM][BLOCKNUM];
    __shared__ T s_B[BLOCKNUM][BLOCKNUM];
    T *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    T *B_begin = d_B + blockIdx.y * BLOCKNUM;

    T sum[COARSENINGFACTOR][COARSENINGFACTOR];
    for (int i = 0; i < COARSENINGFACTOR; i++)
        for (int j = 0; j < COARSENINGFACTOR; j++)
            sum[i][j] = 0;
    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        s_A[threadIdx.x][threadIdx.y] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCKNUM];
        s_A[threadIdx.x + BLOCK][threadIdx.y] = A_begin[(threadIdx.x + BLOCK) * K + threadIdx.y + step * BLOCKNUM];
        s_A[threadIdx.x][threadIdx.y + BLOCK] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCKNUM + BLOCK];
        s_A[threadIdx.x + BLOCK][threadIdx.y + BLOCK] = A_begin[(threadIdx.x + BLOCK) * K + threadIdx.y + step * BLOCKNUM + BLOCK];
        s_B[threadIdx.x][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCKNUM) * N + threadIdx.y];
        s_B[threadIdx.x + BLOCK][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCKNUM + BLOCK) * N + threadIdx.y];
        s_B[threadIdx.x][threadIdx.y + BLOCK] = B_begin[(threadIdx.x + step * BLOCKNUM) * N + threadIdx.y + BLOCK];
        s_B[threadIdx.x + BLOCK][threadIdx.y + BLOCK] = B_begin[(threadIdx.x + step * BLOCKNUM + BLOCK) * N + threadIdx.y + BLOCK];
        __syncthreads();
        for (int i = 0; i < COARSENINGFACTOR; i++)
        {
            for (int j = 0; j < COARSENINGFACTOR; j++)
            {
                int tx = threadIdx.x + i * BLOCK;
                int ty = threadIdx.y + j * BLOCK;
                for (int k = 0; k < BLOCKNUM; k++)
                {
                    sum[i][j] += s_A[tx][k] * s_B[k][ty];
                }
            }
        }
        __syncthreads();
    }

    const int C_m = threadIdx.x + blockIdx.x * BLOCKNUM;
    const int C_n = threadIdx.y + blockIdx.y * BLOCKNUM;
    d_C[C_m * N + C_n] = sum[0][0];
    d_C[C_m * N + C_n + BLOCK] = sum[0][1];
    d_C[(C_m + BLOCK) * N + C_n] = sum[1][0];
    d_C[(C_m + BLOCK) * N + C_n + BLOCK] = sum[1][1];
}
template <class T>
void hgemm_cuda_threadCoarsening(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    const int BLOCK = 16;
    const int COARSENINGFACTOR = 2;
    dim3 Grid((M + BLOCK - 1) / (BLOCK * COARSENINGFACTOR), (N + BLOCK - 1) / (BLOCK * COARSENINGFACTOR));
    dim3 Block(BLOCK, BLOCK);
    hgemm_cuda_threadCoarsening_kernel<T, BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    h_C = d_C;
}