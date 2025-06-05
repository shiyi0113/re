#pragma once
#include <cute/tensor.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename TA, typename TB, typename TC, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void hgemm_cute_naive_kernel(TC *Cptr, const TA *Aptr, const TB *Bptr, int m, int n, int k)
{
    using namespace cute;
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); //(m,k):(k,1) //(81920,256):(256,1)
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(n,k):(k,1) //(256  ,256):(256,1)
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); //(m,n):(n,1) //(81920,256):(256,1)

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    //  gA(kTileM, kTileK, num_tile_k) //(128,64 ,4):(256,1,64)
    //  gB(kTileN, kTileK, num_tile_k) //(128,64 ,4):(256,1,64)
    //  gC(kTileM, kTileN)             //(128,128  ):(256,1)

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)  ((_2,_2,_2),_8 ,_4 ,4):((_1,2048,_8),4096,_16,_64)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)  ((_2,_2)   ,_16,_4 ,4):((_1,_8),2048,_16,_64)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)              ((_2,_2)   ,_8 ,_16)  :((_1,2048),4096,_8)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)  ((_2,_2,_2),_8 ,_4 ):((_1,_2,_4),_32,_8)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)  ((_2,_2)   ,_16,_4 ):((_1,_2)   ,_16,_4)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)  ((_2,_2)   ,_8 ,_16):((_1,_2)   ,_4 ,_32)

    clear(tCrC);

    int num_tile_k = size<2>(gA);

    for (int itile = 0; itile < num_tile_k; ++itile)
    {
        cute::copy(tAgA(_, _, _, itile), tArA);        // g->r
        cute::copy(tBgB(_, _, _, itile), tBrB);        // g->r
        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC); // gemm   C = A * B + C
    }

    cute::copy(tCrC, tCgC); // r->g
}
template <class T>
void hgemm_cute_naive(thrust::host_vector<T> h_A, thrust::host_vector<T> h_B, thrust::host_vector<T> h_C, const int m = 81920, const int n = 256, const int k = 256)
{
    using namespace cute;
    using TA = T;
    using TB = T;
    using TC = T;
    const int kTileN = 128;
    const int kTileM = 128;
    const int kTileK = 64;

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using mma_op = cute::SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        Tile<_32, _16, _16>{}));
    dim3 block(size(MMA{}));
    dim3 grid(n / kTileN, m / kTileM);
    hgemm_cute_naive_kernel<TA, TB, TC, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(
        d_C.data().get(),
        d_A.data().get(),
        d_B.data().get(),
        m, n, k);
    CUTE_CHECK_LAST();
    h_C = d_C;
}