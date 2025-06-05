#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#define FETCH_HALF4(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
template <class T, unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void hgemm_cuda_registerOuter_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    constexpr int NUM_PER_THREAD = COARSENINGFACTOR * COARSENINGFACTOR;
    __shared__ T s_A[BLOCKNUM][BLOCKNUM];
    __shared__ T s_B[BLOCKNUM][BLOCKNUM];
    T *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    T *B_begin = d_B + blockIdx.y * BLOCKNUM;
    T sum[COARSENINGFACTOR][COARSENINGFACTOR];
    T reg_A[COARSENINGFACTOR];
    T reg_B[COARSENINGFACTOR];
    const int tid = ty * blockDim.x + tx;
    const int ntx = tid / 16;
    const int nty = tid % 16;
    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        FETCH_HALF4(s_A[tx][ty * NUM_PER_THREAD]) = FETCH_HALF4(A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM]);
        FETCH_HALF4(s_B[tx][ty * NUM_PER_THREAD]) = FETCH_HALF4(B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD]);
        __syncthreads();
        for (int k = 0; k < BLOCKNUM; k++)
        {
            for (int i = 0; i < COARSENINGFACTOR; i++)
            {
                reg_A[i] = s_A[ntx * COARSENINGFACTOR + i][k];
                reg_B[i] = s_B[k][nty * COARSENINGFACTOR + i];
            }
            for (int i = 0; i < COARSENINGFACTOR; i++)
            {
                for (int j = 0; j < COARSENINGFACTOR; j++)
                {
                    sum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }
    T *C_begin = d_C + blockIdx.x * BLOCKNUM * N + blockIdx.y * BLOCKNUM;
    for (int i = 0; i < COARSENINGFACTOR; i++)
    {
        for (int j = 0; j < COARSENINGFACTOR; j++)
        {
            C_begin[(ntx * COARSENINGFACTOR + i) * N + nty * COARSENINGFACTOR + j] = sum[i][j];
        }
    }
}
template <class T>
void hgemm_cuda_registerOuter(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    const int BLOCK = 16;
    const int COARSENINGFACTOR = 2;
    dim3 Grid((M + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR), (N + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR));
    dim3 Block(BLOCK * COARSENINGFACTOR, BLOCK / COARSENINGFACTOR);
    hgemm_cuda_registerOuter_kernel<T, BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    h_C = d_C;
}