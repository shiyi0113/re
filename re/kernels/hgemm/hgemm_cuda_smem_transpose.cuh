#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#define FETCH_HALF4(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
template <class T, unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void hgemm_cuda_transposeA_kernel(T *d_A, T *d_B, T *d_C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    __shared__ T s_A[BLOCKNUM][BLOCKNUM];
    __shared__ T s_B[BLOCKNUM][BLOCKNUM];
    T *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    T *B_begin = d_B + blockIdx.y * BLOCKNUM;
    T reg_gtos_A[COARSENINGFACTOR];
    T sum[COARSENINGFACTOR][COARSENINGFACTOR];
    T reg_A[COARSENINGFACTOR];
    T reg_B[COARSENINGFACTOR];

    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        for (int i = 0; i < COARSENINGFACTOR; i++)
        {
            FETCH_HALF4(reg_gtos_A[0]) = FETCH_HALF4(A_begin[(tx * COARSENINGFACTOR + i) * K + ty * COARSENINGFACTOR + step * BLOCKNUM]);
            s_A[ty * COARSENINGFACTOR][tx * COARSENINGFACTOR + i] = reg_gtos_A[0];
            s_A[ty * COARSENINGFACTOR + 1][tx * COARSENINGFACTOR + i] = reg_gtos_A[1];
            s_A[ty * COARSENINGFACTOR + 2][tx * COARSENINGFACTOR + i] = reg_gtos_A[2];
            s_A[ty * COARSENINGFACTOR + 3][tx * COARSENINGFACTOR + i] = reg_gtos_A[3];
            FETCH_HALF4(s_B[tx * COARSENINGFACTOR + i][ty * COARSENINGFACTOR]) = FETCH_HALF4(B_begin[(tx * COARSENINGFACTOR + i + step * BLOCKNUM) * N + ty * COARSENINGFACTOR]);
        }
        __syncthreads();
        for (int k = 0; k < BLOCKNUM; k++)
        {

            FETCH_HALF4(reg_A[0]) = FETCH_HALF4(s_A[k][tx * COARSENINGFACTOR + 0]);
            FETCH_HALF4(reg_B[0]) = FETCH_HALF4(s_B[k][ty * COARSENINGFACTOR + 0]);

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
        FETCH_HALF4(C_begin[(tx * COARSENINGFACTOR + i) * N + ty * COARSENINGFACTOR + 0]) = FETCH_HALF4(sum[i][0]);
    }
}
template <class T>
void hgemm_cuda_transposeA(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int M, const int N, const int K)
{
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    const int BLOCK = 16;
    const int COARSENINGFACTOR = 4;
    const int BLOCKSIZE = BLOCK * COARSENINGFACTOR;
    dim3 Grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 Block(BLOCK, BLOCK);
    hgemm_cuda_transposeA_kernel<T, BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    h_C = d_C;
}