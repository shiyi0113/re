#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#define FETCH_HALF4(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
template <class T, unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void hgemm_cuda_usingHalf4_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    constexpr int NUM_PER_THREAD = COARSENINGFACTOR * COARSENINGFACTOR;
    __shared__ T s_A[BLOCKNUM][BLOCKNUM];
    __shared__ T s_B[BLOCKNUM][BLOCKNUM];
    T *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    T *B_begin = d_B + blockIdx.y * BLOCKNUM;
    T sum[NUM_PER_THREAD];
    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        /*
        s_A[tx][ty * NUM_PER_THREAD] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM];
        s_A[tx][ty * NUM_PER_THREAD + 1] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 1];
        s_A[tx][ty * NUM_PER_THREAD + 2] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 2];
        s_A[tx][ty * NUM_PER_THREAD + 3] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 3];
        s_B[tx][ty * NUM_PER_THREAD] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD];
        s_B[tx][ty * NUM_PER_THREAD + 1] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 1];
        s_B[tx][ty * NUM_PER_THREAD + 2] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 2];
        s_B[tx][ty * NUM_PER_THREAD + 3] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 3];
        */
        FETCH_HALF4(s_A[tx][ty * NUM_PER_THREAD]) = FETCH_HALF4(A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM]);
        FETCH_HALF4(s_B[tx][ty * NUM_PER_THREAD]) = FETCH_HALF4(B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD]);
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREAD; i++)
        {
            for (int k = 0; k < BLOCKNUM; k++)
            {
                sum[i] += s_A[tx][k] * s_B[k][ty * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    const int C_m = blockIdx.x * BLOCKNUM + threadIdx.x;
    const int C_n = blockIdx.y * BLOCKNUM + threadIdx.y * NUM_PER_THREAD;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        d_C[C_m * N + C_n + i] = sum[i];
    }
}
template <class T>
void hgemm_cuda_usingHalf4(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    const int BLOCK = 16;
    const int COARSENINGFACTOR = 4;
    dim3 Grid((M + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR), (N + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR));
    dim3 Block(BLOCK * COARSENINGFACTOR, BLOCK / COARSENINGFACTOR);
    hgemm_cuda_usingHalf4_kernel<T, BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    h_C = d_C;
}