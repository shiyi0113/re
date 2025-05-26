#include <cute/tensor.hpp>

#include <core/data.cuh>
#include <core/cublaslt-gemm.cuh>
using namespace cute;

template <class Config>
__global__ void hgemm_cute_multiStage_kernel(void *d_C, const void *d_A, const void *d_B, int m, int n, int k)
{
    using T = typename Config::T;

    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor A = make_tensor(make_gmem_ptr((T *)d_A), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((T *)d_B), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((T *)d_C), make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    Tensor sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    // MMA
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    /**********COPY**********/
    // A G2S
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);
    // B G2S
    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);
    // A S2R
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
    // B S2R
    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    /**********pipline**********/
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    int ntile = k / kTileK;

    for (int istage = 0; istage < kStage - 1; ++istage)
    {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));

        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));
    for (int itile = 0; itile < ntile; ++itile)
    {
        int nk = size<2>(tCrA);
        for (int ik = 0; ik < nk; ++ik)
        {
            int ik_next = (ik + 1) % nk;
            if (ik == nk - 1)
            {
                cp_async_wait<kStage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0)
            {
                if (itile_to_read < ntile)
                {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
        }
    }
    // COPY
    // C R2S
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
    // C S2G
    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gC);
    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    int step = size<3>(tCsC_r2s);

    for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
    {
        for (int j = 0; j < step; ++j)
        {
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

        for (int j = 0; j < step; ++j)
        {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

template <class T_,
          int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 3, int kSmemLayoutCBatch_ = 2>
struct GemmConfig
{
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;
    // MMA
    using mma_op = cute::SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        MMA_EU_RepeatT{},
                                        MMA_P_T{}));
    static constexpr int kThreadNum = size(MMA{});
    // SMEM
    using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{},
                                                make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                                            make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutAtomC = decltype(composition(Swizzle<2, 3, 3>{},
                                                 make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                                             make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));
    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
    /**********COPY**********/
    // AB-g2s
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                              make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                              make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;
    // AB-s2r
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;
    // check
    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}), "C shared memory request is large than A's one pipe");
    // C-r2s
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    // C-s2g
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
                                              make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                              make_layout(make_shape(Int<1>{}, Int<8>{}))));
};

void hgemm_cute_multiStage()
{
    using T = cute::half_t;
    srand(521);
    constexpr int M = 81920;
    constexpr int N = 256;
    constexpr int K = 256;
    constexpr int kTileN = 128;
    constexpr int kTileM = 128;
    constexpr int kTileK = 32;
    constexpr int kStage = 3;

    T *h_A = (T *)malloc(sizeof(T) * M * K);
    T *h_B = (T *)malloc(sizeof(T) * N * K);
    T *h_C = (T *)malloc(sizeof(T) * M * N);
    auto tA = make_tensor(h_A, make_shape(M, K), make_stride(K, 1));
    auto tB = make_tensor(h_B, make_shape(N, K), make_stride(K, 1));
    auto tC = make_tensor(h_C, make_shape(M, N), make_stride(N, 1));
    cpu_rand_data(&tA);
    cpu_rand_data(&tB);
    clear(tC);

    T *d_A;
    T *d_B;
    T *d_C;
    cudaMalloc(&d_A, sizeof(T) * M * K);
    cudaMalloc(&d_B, sizeof(T) * N * K);
    cudaMalloc(&d_C, sizeof(T) * M * N);
    cudaMemcpy(d_A, h_A, sizeof(T) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(T) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(T) * M * N, cudaMemcpyHostToDevice);
    // CPU-Calculate
    T *h_C_CPU = (T *)malloc(sizeof(T) * M * N);
    auto tC_cpu = make_tensor(h_C_CPU, make_shape(M, N), make_stride(N, 1));
    cpu_gemm(&tC_cpu, tA, tB);

    // kernel
    GemmConfig<T, kTileM, kTileN, kTileK, kStage> gemm_config;

    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int shm_size = gemm_config.kShmSize;

    cudaFuncSetAttribute(hgemm_cute_multiStage_kernel<decltype(gemm_config)>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    hgemm_cute_multiStage_kernel<decltype(gemm_config)><<<grid, block, shm_size>>>(d_C, d_A, d_B, M, N, K);
    cudaMemcpy(h_C, d_C, sizeof(T) * M * N, cudaMemcpyDeviceToHost);

    // error throw
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y, grid.x, grid.y, shm_size);

    if (err == cudaSuccess)
    {
        printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }
    else
    {
        printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }

    // compare
    cpu_compare(tC_cpu, tC, 0.1f);

    // print
    auto tile = make_tile(min(8, M), min(8, N));
    auto t32x32 = local_tile(tC, tile, make_coord(0, 0));
    auto t32x32_cpu = local_tile(tC_cpu, tile, make_coord(0, 0));
    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("my kernel:\n");
    print_tensor(t32x32);
    printf("cpu:\n");
    print_tensor(t32x32_cpu);
}