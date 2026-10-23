// MXFP4 A4W4 GEMM for gfx950 (MI350).
//
// Computes out[M,N] = dequant(A) @ dequant(B)^T with MXFP4 (e2m1) 4-bit
// operands packed two-per-byte and per-32-K-block E8M0 scales, consuming the
// production shuffled-input layout (plain row-major A, 16x16-preshuffled B,
// shuffled E8M0 scales -- see aiter/ops/shuffle.py, aiter/utility/fp4_utils.py).
//
// Compute core: the gfx950 scaled MFMA
//   v_mfma_scale_f32_16x16x128_f8f6f4 (fmt=4 -> fp4 e2m1),
// decoded empirically with kernel/mfma_probe.hip:
//   * input fragment: lane L holds row (L%16), K-block g=(L/16); the block's
//     32 fp4 (16 bytes = i32x4) are the operand (fp4 ignores the nominal high
//     128 bits of the 256-bit f8f6f4 slot).
//   * scale is PER-LANE: lane L passes the E8M0 byte of its own (row, block)
//     in the low byte (opsel=0); block g = L/16 so the 4 lane-groups cover the
//     four 32-K blocks of the 128-K MFMA -> correct per-block MXFP4 scaling.
//   * output: acc[i] on lane L == C[(L/16)*4 + i][L%16].
//
// Layout decode (flat offsets):
//   B preshuffle, into the [N, K/2] contiguous buffer:
//     flatB(n,kp) = (n/16)*(KT*512) + (kp/32)*512 + ((kp%32)/16)*256
//                   + (n%16)*16 + (kp%16)         with KT = K/64
//   Scale shuffle, into the scale buffer (row r, K-block b):
//     flatS(r,b) = ((r%32)/16) + ((b%8)/4)*2 + ((r%32)%16)*4
//                  + ((b%8)%4)*64 + (b/8)*256 + (r/32)*32*scaleN_pad
//
// Performance design (what actually moves gfx950 MXFP4 GEMM throughput):
//   * The matrix engine is the limiter, NOT occupancy. Profiling shows the
//     production asm kernel sustains ~56% MFMA-unit utilization at <1 wave/SIMD;
//     occupancy does nothing. What keeps the pipe fed is MFMA ISSUE DENSITY --
//     many independent MFMAs in flight per K-step (deep ILP).
//   * So each wave computes a LARGE tile: up to 128x128 (MSUB=NSUB=8) = 64
//     independent 16x16 MFMAs per 128-K step. That needs 64 f32x4 accumulators
//     = 256 registers, which only fit if they live in the AGPR file -- hence the
//     inline-asm MFMA (mfma_fp4_acc, "+a" accumulator). The builtin keeps
//     accumulators in VGPRs (capped at 256) and cannot reach this tile size.
//   * The K loop is software-pipelined with ping-pong register buffers: this
//     overlaps the next chunk's loads with the current MFMAs AND is required for
//     correctness -- the MFMA reads its source VGPRs over many cycles, opaque to
//     the compiler's hazard model, so a single buffer lets the next load clobber
//     operands still in flight.
//   * The launcher picks a tile per problem (HUGE 128x128 down to SMALL 16x32)
//     so small M*N shapes, which can't fill the GPU with 128x128 tiles, instead
//     get more workgroups.

#include "gemm_a4w4.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <type_traits>
#include <utility>
#include <algorithm>
#include <iostream>

namespace {

typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// Flat offset of weight byte (n, kp) inside the preshuffled B buffer.
__device__ __forceinline__ int64_t flatB(int n, int kp, int KT)
{
    int n_tile = n >> 4;
    int n_in   = n & 15;
    int k_tile = kp >> 5;
    int rem    = kp & 31;
    int kpack  = rem >> 4;
    int k_in   = kp & 15;
    return (int64_t)n_tile * ((int64_t)KT * 512) + (int64_t)k_tile * 512 +
           (int64_t)kpack * 256 + (int64_t)n_in * 16 + k_in;
}

// Flat offset of scale byte for row r, K-block b inside the shuffled buffer.
__device__ __forceinline__ int64_t flatS(int r, int b, int scaleN_pad)
{
    int r32 = r & 31;
    int b8  = b & 7;
    return (int64_t)(r32 >> 4) + (int64_t)(b8 >> 2) * 2 + (int64_t)(r32 & 15) * 4 +
           (int64_t)(b8 & 3) * 64 + (int64_t)(b >> 3) * 256 +
           (int64_t)(r >> 5) * 32 * (int64_t)scaleN_pad;
}

// Load one lane's 32-fp4 (16-byte, 16-byte-aligned) K-block fragment. For the
// fp4 (fmt=4) MFMA the operand is 128 bits = 4 i32 (the instruction ignores the
// high 128 bits of the nominal 256-bit f8f6f4 slot), so we carry just an i32x4.
__device__ __forceinline__ i32x4 load_frag(const uint8_t* p, bool valid)
{
    return valid ? *reinterpret_cast<const i32x4*>(p) : i32x4{0, 0, 0, 0};
}

// Scaled MXFP4 MFMA with the accumulator pinned to the AGPR file ("+a"). Keeping
// the 16 f32x4 accumulators in AGPRs (not VGPRs) frees the capped 256-VGPR pool
// for software-pipeline buffers -- which is what lets the K loop prefetch. The
// asm is non-volatile so the compiler may still interleave loads between MFMAs.
//   D, A, B, C, scaleA, scaleB  (D==C == the accumulator, in place)
__device__ __forceinline__ void mfma_fp4_acc(f32x4& acc, i32x4 a, i32x4 b,
                                             int sa, int sb)
{
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 "
        "op_sel_hi:[0,0,0] cbsz:4 blgp:4"
        : "+a"(acc)
        : "v"(a), "v"(b), "v"(sa), "v"(sb));
}

// 2D register blocking: each wave computes an (16*MSUB) x (16*NSUB) output tile.
// Per K-step it loads MSUB A fragments (+scales) and NSUB B fragments (+scales)
// and issues MSUB*NSUB MFMAs, so A traffic is cut by NSUB and B traffic by MSUB.
//
// MSUB/NSUB are a compile-time tile shape. Bigger tiles give more A/B reuse
// (higher peak on large shapes) but cost more VGPRs -> lower occupancy; smaller
// tiles raise occupancy and the workgroup count (better for small M*N). The
// launcher instantiates a few shapes and picks one per problem (see dispatch).

// One wave (64 lanes) computes a (16*MSUB) x (16*NSUB) output tile.
//
// launch_bounds min-waves/EU: the double-buffered BIG tile (16 MFMAs -> 64 AGPR
// accumulators + 2x fragment buffers) needs more than the 128-reg budget of
// occupancy 4, so permit occupancy 3 (512/3 regs) to avoid VGPR spills. MID/
// SMALL are small enough that 1 (no constraint) lets them reach occupancy 5-7.
template <int MSUB, int NSUB>
__global__ void __launch_bounds__(64, MSUB* NSUB >= 32 ? 1 : (MSUB* NSUB >= 16 ? 2 : 1))
gemm_a4w4_mfma_kernel(const uint8_t* __restrict__ A,
                                      const uint8_t* __restrict__ B,
                                      const uint8_t* __restrict__ AScale,
                                      const uint8_t* __restrict__ BScale,
                                      __hip_bfloat16* __restrict__ out,
                                      int M, int N, int K, int Mpad,
                                      int scaleN_pad)
{
    constexpr int MTILE = 16 * MSUB;
    constexpr int NTILE = 16 * NSUB;
    const int lane  = threadIdx.x;   // 0..63
    const int row16 = lane & 15;     // fragment row/col index
    const int g     = lane >> 4;     // K-block group (0..3)
    const int m0    = blockIdx.y * MTILE;
    const int n0    = blockIdx.x * NTILE;

    const int Kp      = K >> 1;       // packed bytes per row
    const int KT      = K / 64;       // k_tiles for B preshuffle
    const int Kchunks = K / 128;      // 128-fp4 K steps (one MFMA each)

    f32x4 acc[MSUB][NSUB];
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Load chunk kc's A/B fragments + E8M0 scales into the given buffers.
    auto load = [&](int kc, i32x4 (&af)[MSUB], int (&as)[MSUB],
                    i32x4 (&bf)[NSUB], int (&bs)[NSUB]) {
        const int kblk     = 4 * kc + g;
        const int ktile    = 2 * kc + (g >> 1);
        const int kpack    = g & 1;
        const int kpstart  = ktile * 32 + kpack * 16;
        const int64_t aoff = (int64_t)(kc * 64 + g * 16);
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
        {
            int am = m0 + mi * 16 + row16;
            af[mi] = load_frag(A + (int64_t)am * Kp + aoff, am < M);
            as[mi] = (am < M) ? AScale[flatS(am, kblk, scaleN_pad)] : 127;
        }
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
        {
            int  bn = n0 + ni * 16 + row16;
            bool vb = bn < N;
            bf[ni] = load_frag(B + flatB(bn, kpstart, KT), vb);
            bs[ni] = vb ? BScale[flatS(bn, kblk, scaleN_pad)] : 127;
        }
    };

    auto mma = [&](i32x4 (&af)[MSUB], int (&as)[MSUB], i32x4 (&bf)[NSUB],
                   int (&bs)[NSUB]) {
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int ni = 0; ni < NSUB; ++ni)
                mfma_fp4_acc(acc[mi][ni], af[mi], bf[ni], as[mi], bs[ni]);
    };

    // Software-pipelined K loop with statically named ping-pong buffers (A/B).
    // Besides overlapping loads with compute, double-buffering is REQUIRED for
    // correctness here: the MFMA reads its source VGPRs over many cycles, but
    // the inline asm is opaque to the compiler's hazard model, so a single
    // buffer would let the next chunk's loads clobber operands still being read.
    // Separate buffers keep each chunk's operands alive until its MFMAs finish.
    i32x4 afA[MSUB], bfA[NSUB], afB[MSUB], bfB[NSUB];
    int   asA[MSUB], bsA[NSUB], asB[MSUB], bsB[NSUB];
    load(0, afA, asA, bfA, bsA);

    int kc = 0;
    for (; kc + 1 < Kchunks; kc += 2)
    {
        load(kc + 1, afB, asB, bfB, bsB);   // prefetch odd
        mma(afA, asA, bfA, bsA);            // compute even
        if (kc + 2 < Kchunks)
            load(kc + 2, afA, asA, bfA, bsA);  // prefetch next even
        mma(afB, asB, bfB, bsB);            // compute odd
    }
    if (kc < Kchunks)                        // odd Kchunks tail
        mma(afA, asA, bfA, bsA);

    // acc[mi][ni][i] on lane L == C[(L/16)*4 + i][L%16] of subtile (mi, ni).
#pragma unroll
    for (int ni = 0; ni < NSUB; ++ni)
    {
        const int on = n0 + ni * 16 + row16;
        if (on >= N)
            continue;
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int om = m0 + mi * 16 + g * 4 + i;
                if (om < M)
                    out[(int64_t)om * N + on] =
                        __float2bfloat16(acc[mi][ni][i]);
            }
    }
}

// ---------------------------------------------------------------------------
// Workgroup-cooperative kernel (modeled on the production asm f4gemm 160x256).
//
// One workgroup = WAVES waves laid out 1 x WAVES along N; each wave computes a
// (16*MSUB) x (16*NSUB) tile, so the workgroup tile is (16*MSUB) x
// (16*NSUB*WAVES). Compared to the single-wave kernel above:
//   * A is shared by all waves of the workgroup through LDS. It is streamed
//     with direct-to-LDS buffer loads (no VGPR staging) into a 4-deep ring,
//     one K=256 stage (MTILE rows x 128 B) per buffer, 2 stages ahead of
//     compute. The 16 B chunks of each row are XOR-swizzled by (row & 7) so
//     the ds_read_b128 fragment reads (16 lanes on 16 rows, same K chunk) do
//     not bank-conflict.
//   * B (16x16 preshuffled) still goes straight from global into VGPRs -- each
//     wave owns distinct N columns, so there is nothing to share -- 2 stages
//     ahead (1 for the tiles that cannot afford a third B register set).
//   * E8M0 scales are fetched one dword per lane per (16-row pair, K=256 stage):
//     the shuffled scale layout packs [tile parity][K-128 half] into the four
//     bytes, which the MFMA selects for free via op_sel (byte = 2*khalf + par).
//   * Loads, LDS fragment reads and the stage barrier are threaded through the
//     MFMA stream at fixed points (see `step`), as the asm kernel does. All
//     global loads and the MFMAs are inline asm; the loop's vmcnt waits are
//     explicit and the accumulators are pinned to AGPRs.
//   * Persistent (one workgroup per CU at 1 wave/SIMD) with an XCD-aware tile
//     order; the pipeline runs across tile boundaries.
//   * The MFMA computes C^T (operands swapped) so each lane holds 4 contiguous
//     outputs of a row; the epilogue stages them through LDS into 16-byte
//     row-contiguous stores.
// Requirements (checked by the launcher): K % 1024 == 0 (K=256 stage count a
// multiple of the LDS ring), N % 16 == 0, MSUB/NSUB even, NSUB in {4, 6, 8}.
// ---------------------------------------------------------------------------

// Scaled fp4 MFMA with compile-time E8M0 byte selection (byte = op_sel_hi*2 +
// op_sel per operand), as inline asm with the accumulator pinned to AGPRs
// ("+a"). With the builtin, the register allocator is free to move
// accumulators between AGPRs and VGPRs; for the big tiles (up to 256
// accumulator registers) it does so inside the K loop, costing hundreds of
// accvgpr copies per stage. Hazard note: the asm is opaque to the hazard
// recognizer, but every accumulator is touched again only after many other
// MFMAs, and its fragment/scale sources are rewritten only by loads issued
// long after their last MFMA.
#define MBS_MFMA_FP4(OS0, OS1, OH0, OH1)                                                   \
    asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 op_sel:[" #OS0 \
                 "," #OS1 ",0] op_sel_hi:[" #OH0 "," #OH1 ",0] cbsz:4 blgp:4"              \
                 : "+a"(acc)                                                              \
                 : "v"(a), "v"(b), "v"(sa), "v"(sb))
template <int OA, int OB>
__device__ __forceinline__ void mfma_fp4_sel(f32x4& acc, i32x4 a, i32x4 b, int sa, int sb)
{
    static_assert(OA >= 0 && OA < 4 && OB >= 0 && OB < 4, "scale byte select");
    if constexpr (OA == 0 && OB == 0) MBS_MFMA_FP4(0, 0, 0, 0);
    else if constexpr (OA == 1 && OB == 0) MBS_MFMA_FP4(1, 0, 0, 0);
    else if constexpr (OA == 2 && OB == 0) MBS_MFMA_FP4(0, 0, 1, 0);
    else if constexpr (OA == 3 && OB == 0) MBS_MFMA_FP4(1, 0, 1, 0);
    else if constexpr (OA == 0 && OB == 1) MBS_MFMA_FP4(0, 1, 0, 0);
    else if constexpr (OA == 1 && OB == 1) MBS_MFMA_FP4(1, 1, 0, 0);
    else if constexpr (OA == 2 && OB == 1) MBS_MFMA_FP4(0, 1, 1, 0);
    else if constexpr (OA == 3 && OB == 1) MBS_MFMA_FP4(1, 1, 1, 0);
    else if constexpr (OA == 0 && OB == 2) MBS_MFMA_FP4(0, 0, 0, 1);
    else if constexpr (OA == 1 && OB == 2) MBS_MFMA_FP4(1, 0, 0, 1);
    else if constexpr (OA == 2 && OB == 2) MBS_MFMA_FP4(0, 0, 1, 1);
    else if constexpr (OA == 3 && OB == 2) MBS_MFMA_FP4(1, 0, 1, 1);
    else if constexpr (OA == 0 && OB == 3) MBS_MFMA_FP4(0, 1, 0, 1);
    else if constexpr (OA == 1 && OB == 3) MBS_MFMA_FP4(1, 1, 0, 1);
    else if constexpr (OA == 2 && OB == 3) MBS_MFMA_FP4(0, 1, 1, 1);
    else MBS_MFMA_FP4(1, 1, 1, 1);
}
#undef MBS_MFMA_FP4

// Two floats -> packed bf16x2 (RNE), lowered to v_cvt_pk_bf16_f32 on gfx950.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(lo)) |
           ((uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(hi)) << 16);
}

template <int... Is, typename F>
__device__ __forceinline__ void static_for_impl(std::integer_sequence<int, Is...>, F&& f)
{
    (f(std::integral_constant<int, Is>{}), ...);
}
template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f)
{
    static_for_impl(std::make_integer_sequence<int, N>{}, f);
}

template <int MSUB, int NSUB, int WAVES>
__global__ void __launch_bounds__(64 * WAVES, 1)
gemm_a4w4_wg_kernel(const uint8_t* __restrict__ A,
                    const uint8_t* __restrict__ B,
                    const uint8_t* __restrict__ AScale,
                    const uint8_t* __restrict__ BScale,
                    __hip_bfloat16* __restrict__ out,
                    int M, int N, int K, int scaleN_pad)
{
    static_assert(MSUB % 2 == 0 && NSUB % 2 == 0, "scale dwords cover tile pairs");
    constexpr int MTILE   = 16 * MSUB;
    constexpr int WNTILE  = 16 * NSUB;            // per-wave N extent
    constexpr int NTILE   = WNTILE * WAVES;   // hangy: warp沿着N方向排列
    constexpr int STAGE_B = MTILE * 128;          // LDS bytes per K=256 A stage
    constexpr int A_LDS_PER_WAVE = MTILE / 8 / WAVES;  // 1 KB (8 rows) each
    static_assert((MTILE / 8) % WAVES == 0, "A stage must split evenly across waves");
    // Epilogue staging rows: WNTILE bf16 + 16 B pad, in ECPR 16-byte chunks.
    constexpr int ECPR   = WNTILE / 8;
    constexpr int EROW_B = WNTILE * 2 + 16;
    static_assert((32 * ECPR) % 64 == 0, "epilogue read-back covers whole waves");

    // Ring of RING LDS stage buffers. A (via LDS) is prefetched PF stages ahead,
    // B and the scales (straight into registers) PF_B stages ahead. Each B
    // stage in flight costs a register set (2*NSUB*4 VGPRs): the 256-row and
    // the wide-N tiles, which run 2*MSUB*NSUB >= 128 MFMAs per stage, keep B
    // one stage ahead; the rest measured better at two.
    constexpr int RING = 4;
    constexpr int PF   = 2;
    constexpr int PF_B = (NSUB <= 4 && MSUB < 16) ? 2 : 1;
    // Step s refills buffer (s+PF) % RING before its barrier. With RING = PF+1
    // that is stage s-1's buffer, whose reads all issue before step s-1's
    // barrier and are retired there (WAIT_KEEP includes lgkmcnt(0)).
    static_assert(RING >= PF + 1, "refill target must be clear of readers");
    static_assert(PF_B <= PF, "B is never further ahead than A");
    __shared__ __attribute__((aligned(16))) uint8_t smem0[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem1[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem2[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem3[STAGE_B];
    // Epilogue staging, separate from the ring (which already holds the next
    // tile's prefetched stages): 32 rows per wave.
    __shared__ __attribute__((aligned(16))) uint8_t sepi[WAVES][32 * EROW_B];
    // Distinct arrays, statically selected per (unrolled) ring slot.
    auto lds_buf = [&](auto u) __attribute__((always_inline)) -> uint8_t* {
        constexpr int i = decltype(u)::value % RING;
        if constexpr (i == 0) return smem0;
        else if constexpr (i == 1) return smem1;
        else if constexpr (i == 2) return smem2;
        else return smem3;
    };

    const int lane  = threadIdx.x & 63;
    // readfirstlane: the compiler cannot tell threadIdx.x >> 6 is wave-uniform;
    // made explicit, every wave-derived tile base lives in SGPRs, not VGPRs.
    const int wave  = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int row16 = lane & 15;
    const int g     = lane >> 4;

    const int Kp      = K >> 1;
    const int KT      = K / 64;
    const int nstages = K / 256;

    // ---- persistent tile schedule ---------------------------------------------
    // One resident workgroup per CU walks tiles id = blockIdx.x + i*gridDim.x.
    // The software pipeline runs straight across tile boundaries: the next
    // tile's first stages are prefetched under the current tile's last MFMAs,
    // so only the very first tile pays the load latency before compute starts.
    //
    // Rasterization: workgroups are dispatched round-robin over the XCDs (each
    // with a private L2), so a plain row-major order gives every XCD a
    // scattered set of tiles with little B reuse. Remap so each XCD walks a
    // contiguous range of tile ids, ordered in GROUP_M-tall column groups: the
    // ~32 workgroups resident on one XCD then cover a GROUP_M x 8 block, sharing
    // each B tile GROUP_M times and each A tile ~8 times in L2. (gridDim.x is a
    // multiple of NXCD, so id % NXCD is the XCD of the workgroup for every i.)
    constexpr int NXCD = 8, GROUP_M = 4;
    const int num_m = (M + MTILE - 1) / MTILE;
    const int num_n = (N + NTILE - 1) / NTILE;
    const int total = num_m * num_n;

    // Load offsets = per-lane part (tile-independent, below) + per-tile part
    // (wave-uniform, kept in SGPRs and folded into soffset). Buffer loads
    // bounds-check voffset + soffset against the resource size, so everything
    // past M / N (rows, subtiles, scale blocks) reads 0 in hardware.
    struct TileOff
    {
        int m0, n0;  // n0 includes this wave's column offset
        int a, b, as, bs;
    };
    // A -> LDS: wave instruction j fills rows [8*grp, 8*grp+8) of the stage;
    // lane writes LDS slot (row = lane/8, phys chunk = lane%8) and so fetches
    // logical chunk phys ^ (row & 7). j advances 8 rows via soffset.
    const int a_lane = (lane >> 3) * Kp + (((lane & 7) ^ (lane >> 3)) * 16);
    // B fragment (n, K-128 chunk kc): 1 KB contiguous per 16-row subtile.
    const int b_lane = g * 256 + row16 * 16;
    // Scale dword of a 32-row block-stage: the shuffled layout is lane-major.
    const int s_lane = lane * 4;
    auto tile_off = [&](int id) __attribute__((always_inline)) {
        id = min(id, total - 1);  // past-the-end prefetches: any valid tile
        if (id < total / NXCD * NXCD)  // ragged tail keeps its natural id
            id = (id % NXCD) * (total / NXCD) + id / NXCD;
        const int group   = id / (GROUP_M * num_n);
        const int first_m = group * GROUP_M;
        const int gsize   = min(num_m - first_m, GROUP_M);
        TileOff t;
        t.m0 = (first_m + (id % (GROUP_M * num_n)) % gsize) * MTILE;
        t.n0 = (id % (GROUP_M * num_n)) / gsize * NTILE + wave * WNTILE;
        t.a  = (t.m0 + wave * A_LDS_PER_WAVE * 8) * Kp;
        t.b  = (t.n0 >> 4) * KT * 512;
        t.as = (t.m0 >> 5) * 32 * scaleN_pad;  // block holding subtiles (2p, 2p+1)
        t.bs = (t.n0 >> 5) * 32 * scaleN_pad;
        return t;
    };

    constexpr int RSRC_W3 = 0x00020000;
    const int Mpad32 = (M + 31) & ~31;
    const int Npad32 = (N + 31) & ~31;
    // Plain SGPR-quad resources for the inline-asm loads below.
    auto rsrc_of = [&](const void* p, int bytes) __attribute__((always_inline)) {
        const uint64_t a = reinterpret_cast<uint64_t>(p);
        return i32x4{(int)a, (int)(a >> 32) & 0xffff, bytes, RSRC_W3};
    };
    const i32x4 a_rsrc  = rsrc_of(A, M * Kp);
    const i32x4 b_rsrc  = rsrc_of(B, N * Kp);
    const i32x4 as_rsrc = rsrc_of(AScale, Mpad32 * scaleN_pad);
    const i32x4 bs_rsrc = rsrc_of(BScale, Npad32 * scaleN_pad);
    // readfirstlane at the use site: an "s" operand whose value the compiler
    // happens to hold in VGPRs is otherwise emitted as VGPRs.
    auto sq = [](const i32x4& r) __attribute__((always_inline)) {
        return i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
                     __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
    };
    const auto o_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        (void*)out, 0, M * N * (int)sizeof(__hip_bfloat16), RSRC_W3);

    f32x4 acc[MSUB][NSUB];
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Global traffic of one pipeline step, as LOADS_PER_STAGE individually
    // addressable ops (k), in issue order: B fragments, B scales, A scales
    // (into the given registers), then A rows into LDS buffer `lds`.
    //
    // All of it is inline asm, and every vmcnt wait in the K loop is explicit
    // (see W_START / W_MID). The LDS DMA cannot be left to the compiler (it
    // drains vmcnt(0) before every LDS read it cannot prove disjoint), and once
    // it is invisible, compiler-placed waits for B miscount either way --
    // forcing the next stage's A DMA to land early. The loaded registers are
    // only read by the (asm) MFMAs, after the explicit wait.
    constexpr int NB              = 2 * NSUB;
    constexpr int LOADS_PER_STAGE = A_LDS_PER_WAVE + NB + MSUB / 2 + NSUB / 2;
    // op ranges: [0, KBS) B, [KBS, KAS) B scales, [KAS, KA) A scales, [KA, L) A DMA
    constexpr int KBS = NB, KAS = KBS + NSUB / 2, KA = KAS + MSUB / 2;
    const uint32_t lds_wave_off   = wave * A_LDS_PER_WAVE * 1024;
    const int a_jstride           = 8 * Kp;  // A DMA op j covers rows 8j..8j+7
    // A ops load stage `st` of tile `to`; B/scale ops load stage `stb` of tile
    // `tob`.
    auto issue_one = [&](auto kc, const TileOff& to, int st, uint8_t* lds, const TileOff& tob,
                         int stb, i32x4 (&bf)[NSUB][2], int (&as)[MSUB / 2],
                         int (&bs)[NSUB / 2]) __attribute__((always_inline)) {
        constexpr int k = decltype(kc)::value;
        // clang: a capture used only in a dependent branch is missed
        (void)a_jstride, (void)a_rsrc, (void)b_rsrc, (void)as_rsrc, (void)bs_rsrc;
        (void)a_lane, (void)b_lane, (void)s_lane;
        if constexpr (k < KBS)
        {
            constexpr int ni = k / 2, h = k % 2;
            asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen"
                         : "=v"(bf[ni][h])
                         // Tile part in voffset, not soffset: with it in soffset the
                         // big-tile builds fault (cause not pinned down).
                         : "v"(b_lane + tob.b), "s"(sq(b_rsrc)),
                           "s"(__builtin_amdgcn_readfirstlane(ni * KT * 512 + stb * 2048 +
                                                              h * 1024)));
        }
        else if constexpr (k < KAS)
        {
            constexpr int p = k - KBS;
            asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                         : "=v"(bs[p])
                         : "v"(s_lane), "s"(sq(bs_rsrc)),
                           "s"(__builtin_amdgcn_readfirstlane(tob.bs + p * 32 * scaleN_pad +
                                                              stb * 256)));
        }
        else if constexpr (k < KA)
        {
            constexpr int p = k - KAS;
            asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                         : "=v"(as[p])
                         : "v"(s_lane), "s"(sq(as_rsrc)),
                           "s"(__builtin_amdgcn_readfirstlane(tob.as + p * 32 * scaleN_pad +
                                                              stb * 256)));
        }
        else
        {
            constexpr int j = k - KA;
            const uint32_t lds_base =
                (uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)lds + lds_wave_off;
            asm volatile("s_mov_b32 m0, %0\n\t"
                         "buffer_load_dwordx4 %1, %2, %3 offen lds"
                         :
                         : "s"(__builtin_amdgcn_readfirstlane(lds_base + j * 1024)),
                           "v"(a_lane), "s"(sq(a_rsrc)),
                           "s"(__builtin_amdgcn_readfirstlane(to.a + st * 128 + j * a_jstride))
                         : "memory", "m0");
        }
    };

    // One pipeline step (stage s), as one flat MFMA stream of T = 2*MSUB*NSUB:
    //   * the A loads of stage s+PF and B/scale loads of stage s+PF_B are
    //     threaded through the first T_LOAD MFMAs,
    //     ~1 per T_LOAD/LOADS_PER_STAGE MFMAs as the asm kernel does (issuing
    //     ~20 loads back to back stalls the in-order stream on the VMEM queue);
    //   * fragment f = h*MSUB + mi (first used by MFMA f*NSUB) is ds_read
    //     FRAG_AHEAD fragments ahead of its use;
    //   * at MFMA T_LOAD the wave waits for stage s+1's loads and barriers, then
    //     reads stage s+1's first FRAG_AHEAD fragments (`head`) under the last
    //     MFMAs of stage s. The stage boundary then has no barrier + LDS-latency
    //     bubble in front of the first MFMA.
    // sched_barriers pin this interleave, which the compiler would re-cluster.
    constexpr int T          = 2 * MSUB * NSUB;
    constexpr int FRAG_AHEAD = MSUB >= 16 ? 2 : 3;
    constexpr int NFRAG      = 2 * MSUB;
    constexpr int T_LOAD     = T - FRAG_AHEAD * NSUB;
    static_assert(T_LOAD >= LOADS_PER_STAGE, "need at least one MFMA per load");
    // vmcnt bookkeeping (vmcnt retires in issue order; each step issues its
    // B-side ops first, its A DMA last, all before T_LOAD):
    //  * W_START, before step s's first MFMA: its B/scale registers (issued in
    //    step s-PF_B) have landed once at most the newer loads are left: that
    //    step's A DMA plus all of every later step.
    //  * W_MID, at the mid-step barrier: the A DMA for stage s+1 (the tail of
    //    step s+1-PF) has landed once only the later steps' loads are left.
    //  * A tile's epilogue issues EPI_STORES buffer stores (vmcnt counts them
    //    too) between steps, so the first step of the next tile allows those.
    constexpr int EPI_STORES = (MSUB / 2) * (32 * ECPR / 64);
    constexpr int W_START    = (LOADS_PER_STAGE - KA) + LOADS_PER_STAGE * (PF_B - 1);
    constexpr int W_START_T  = W_START + EPI_STORES;
    constexpr int VMCNT_KEEP = LOADS_PER_STAGE * (PF - 1);  // W_MID
    static_assert(VMCNT_KEEP <= 63 && W_START_T <= 63, "vmcnt is 6 bits");
    auto vm_wait = [](auto nc) __attribute__((always_inline)) {
        constexpr int n = decltype(nc)::value;
        __builtin_amdgcn_s_waitcnt((n & 15) | (7 << 4) | (15 << 8) | ((n >> 4) << 14));
    };
    // s_waitcnt simm16 (gfx9): vmcnt[3:0] | expcnt[6:4] | lgkmcnt[11:8] | vmcnt[15:14]
    // W_MID also retires this wave's fragment reads (lgkmcnt(0)): the buffer
    // they read is refilled right after the barrier, see RING.
    constexpr int WAIT_KEEP = (VMCNT_KEEP & 15) | (7 << 4) | (0 << 8) | ((VMCNT_KEEP >> 4) << 14);
    constexpr int WAIT_ALL  = (7 << 4) | (15 << 8);

    auto frag_at = [&](const uint8_t* lds, int f) __attribute__((always_inline)) {
        const int h = f / MSUB, mi = f % MSUB;
        const int row = mi * 16 + row16;
        const int chk = (h * 4 + g) ^ (row & 7);
        return *reinterpret_cast<const i32x4*>(lds + row * 128 + chk * 16);
    };
    i32x4 head[FRAG_AHEAD];

    auto step = [&](const uint8_t* lds, const uint8_t* next_lds, i32x4 (&bf)[NSUB][2],
                    int (&as)[MSUB / 2], int (&bs)[NSUB / 2], const TileOff& nto, int nst,
                    uint8_t* nlds, const TileOff& ntob, int nstb, i32x4 (&nbf)[NSUB][2],
                    int (&nas)[MSUB / 2], int (&nbs)[NSUB / 2], bool after_epi) __attribute__((always_inline)) {
        i32x4 af[NFRAG];
        static_for<FRAG_AHEAD>([&](auto f) __attribute__((always_inline)) { af[f] = head[f]; });
        // This stage's B/scale registers must have landed (see W_START).
        if (after_epi)
            vm_wait(std::integral_constant<int, W_START_T>{});
        else
            vm_wait(std::integral_constant<int, W_START>{});
        __builtin_amdgcn_sched_barrier(0);
        static_for<T>([&](auto tc) __attribute__((always_inline)) {
            constexpr int t = decltype(tc)::value;
            if constexpr (t % NSUB == 0 && t / NSUB + FRAG_AHEAD < NFRAG)
            {
                af[t / NSUB + FRAG_AHEAD] = frag_at(lds, t / NSUB + FRAG_AHEAD);
                __builtin_amdgcn_sched_barrier(0);
            }
            // load k goes before MFMA floor(k*T_LOAD/L)
            constexpr int k = (t * LOADS_PER_STAGE + T_LOAD - 1) / T_LOAD;
            if constexpr (t < T_LOAD && k < LOADS_PER_STAGE && k * T_LOAD / LOADS_PER_STAGE == t)
            {
                issue_one(std::integral_constant<int, k>{}, nto, nst, nlds, ntob, nstb, nbf, nas,
                          nbs);
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr (t == T_LOAD)
            {
                __builtin_amdgcn_s_waitcnt(WAIT_KEEP);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr (t >= T_LOAD && (t - T_LOAD) % NSUB == 0)
            {
                head[(t - T_LOAD) / NSUB] = frag_at(next_lds, (t - T_LOAD) / NSUB);
                __builtin_amdgcn_sched_barrier(0);
            }
            constexpr int h = t / (MSUB * NSUB), mi = (t / NSUB) % MSUB, ni = t % NSUB;
            // Operands swapped (B first): computes the C^T subtile, so lane L
            // ends up holding row L%16, columns (L/16)*4..+3 -- 4 contiguous
            // outputs of one row, which the epilogue packs into 8-byte writes.
            mfma_fp4_sel<2 * h + (ni & 1), 2 * h + (mi & 1)>(
                acc[mi][ni], bf[ni][h], af[t / NSUB], bs[ni / 2], as[mi / 2]);
            __builtin_amdgcn_sched_barrier(0);
        });
    };

    // Epilogue: acc[mi][ni][i] on lane L == C[mi*16 + L%16][ni*16 + (L/16)*4 + i].
    // Stage the wave's tile through LDS 32 rows at a time and write it back as
    // whole row segments with 16-byte stores, instead of 2-byte scattered
    // stores. LDS rows are padded by 16 B (EROW_B) so the 16 lanes writing 16
    // rows of one column spread over banks. Wave-private: no barrier needed.
    // Rows >= M fall outside the buffer and are dropped by the buffer store;
    // chunks past N (N % WNTILE != 0) are masked explicitly.
    auto epilogue = [&](const TileOff& to) __attribute__((always_inline)) {
        // Opaque copy of the lane id: everything below derives from it, so the
        // compiler cannot hoist the per-lane epilogue addresses out of the tile
        // loop -- hoisted, they stay live across the K loop and get spilled,
        // and every spill reload drains the whole prefetch pipeline (vmcnt(0)).
        int lane;
        asm volatile("v_mov_b32 %0, %1" : "=v"(lane) : "v"(threadIdx.x & 63));
        const int row16 = lane & 15, g = lane >> 4;
        uint8_t* const etile = sepi[__builtin_amdgcn_readfirstlane(threadIdx.x >> 6)];
#pragma unroll
        for (int mp = 0; mp < MSUB / 2; ++mp)
        {
            // Fence each 32-row chunk: unfenced, the scheduler hoists every
            // AGPR->VGPR accumulator read to the top, needing MTILE*16 VGPRs.
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int ml = 0; ml < 2; ++ml)
#pragma unroll
                for (int ni = 0; ni < NSUB; ++ni)
                {
                    const f32x4 c = acc[2 * mp + ml][ni];
                    const int row = ml * 16 + row16;
                    *reinterpret_cast<uint2*>(etile + row * EROW_B + (ni * 16 + g * 4) * 2) =
                        uint2{pack_bf16x2(c[0], c[1]), pack_bf16x2(c[2], c[3])};
                }
            // Read back 16-byte chunks, 64 per instruction, row-major over the
            // 32 x ECPR chunk grid of this row block.
            const int sbase = __builtin_amdgcn_readfirstlane(((to.m0 + mp * 32) * N + to.n0) * 2);
#pragma unroll
            for (int j = 0; j < 32 * ECPR / 64; ++j)
            {
                const int idx = j * 64 + lane;
                const int row = idx / ECPR, chk = idx % ECPR;
                const i32x4 v = *reinterpret_cast<const i32x4*>(etile + row * EROW_B + chk * 16);
                // Unconditional (masked lanes get an out-of-range offset, which
                // the buffer store drops): the K loop's vmcnt bookkeeping counts
                // exactly EPI_STORES stores per epilogue.
                __builtin_amdgcn_raw_buffer_store_b128(
                    __builtin_bit_cast(__attribute__((ext_vector_type(4))) unsigned int, v), o_rsrc,
                    to.n0 + chk * 8 < N ? (row * N + chk * 8) * 2 : 0x7fffffff, sbase, 0);
            }
        }
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int ni = 0; ni < NSUB; ++ni)
                acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};
    };

    // Software pipeline: stage s's A loads are issued while stages s-PF..s-1
    // compute (B: PF_B), the stage index running on across tiles
    // (stage >= nstages means stage - nstages of the next tile). Unrolled by
    // RING so every LDS buffer and B/scale register set is statically named;
    // nstages % RING == 0, so every tile starts on ring slot 0. The barrier
    // inside step s both publishes every wave's LDS DMA for stage s+1 and
    // guarantees stage s-1 is fully read before step s+1 refills its buffer.
    // Prefetches past the last tile still issue (for a clamped valid tile)
    // rather than branch: a conditional issue would keep every ring register
    // set live across the loop and spill. They land in a buffer / register set
    // nobody reads.
    i32x4 bf[RING][NSUB][2];
    int as[RING][MSUB / 2], bs[RING][NSUB / 2];

    int id      = blockIdx.x;
    TileOff cur = tile_off(id);
    TileOff nxt = tile_off(id + gridDim.x);
    // Prologue, laid out as the steps "-PF..-1" would issue it so the vmcnt
    // bookkeeping is identical: step -PF+u issues A stage u, B stage u+PF_B-PF.
    static_for<PF>([&](auto u) __attribute__((always_inline)) {
        constexpr int sb = u + PF_B - PF;
        static_for<LOADS_PER_STAGE>([&](auto k) __attribute__((always_inline)) {
            if constexpr (decltype(k)::value >= KA || sb >= 0)  // skip B ops of stages < 0
                issue_one(k, cur, u, lds_buf(u), cur, sb, bf[sb < 0 ? 0 : sb],
                          as[sb < 0 ? 0 : sb], bs[sb < 0 ? 0 : sb]);
        });
    });
    // Stage 0 must have landed before its head fragments are read; later
    // stages do this mid-step.
    __builtin_amdgcn_s_waitcnt(WAIT_KEEP);
    __syncthreads();
    static_for<FRAG_AHEAD>([&](auto f) __attribute__((always_inline)) {
        head[f] = frag_at(smem0, f);
    });
    bool after_epi = false;  // first step of a tile that follows an epilogue
    for (; id < total; id += gridDim.x)
    {
        for (int st = 0; st < nstages; st += RING)
        {
            static_for<RING>([&](auto u) __attribute__((always_inline)) {
                constexpr int nu  = (u + PF) % RING;
                constexpr int nub = (u + PF_B) % RING;
                const int nst     = st + u + PF;
                const int nstb    = st + u + PF_B;
                const bool wrap   = nst >= nstages;   // uniform
                const bool wrapb  = nstb >= nstages;  // uniform
                const TileOff nto{0, 0, wrap ? nxt.a : cur.a, 0, 0, 0};
                const TileOff ntob{0, 0, 0, wrapb ? nxt.b : cur.b, wrapb ? nxt.as : cur.as,
                                   wrapb ? nxt.bs : cur.bs};
                step(lds_buf(u), lds_buf(std::integral_constant<int, u + 1>{}), bf[u], as[u],
                     bs[u], nto, wrap ? nst - nstages : nst,
                     lds_buf(std::integral_constant<int, nu>{}), ntob,
                     wrapb ? nstb - nstages : nstb, bf[nub], as[nub], bs[nub],
                     u == 0 && st == 0 && after_epi);
            });
        }
        epilogue(cur);
        after_epi = true;
        cur = nxt;
        nxt = tile_off(id + 2 * gridDim.x);
    }
    __builtin_amdgcn_s_waitcnt(WAIT_ALL);  // retire the overrun LDS DMA
}

}  // namespace

// Persistent launch of the workgroup kernel: one workgroup per CU (1 wave/SIMD).
template <int MS, int NS>
void launch_wg(const uint8_t* A, const uint8_t* B, const uint8_t* AScale,
               const uint8_t* BScale, void* out, int M, int N, int K, int scaleN_pad,
               hipStream_t stream)
{
    constexpr int WV = 4;  // hangy: 一个tile的size为 (16 * MS) * (16 * NS * WV(4))? 也就是说一个warp的tile是(16 * MS) * (16 * NS)
    const int tiles  = ((N + 16 * NS * WV - 1) / (16 * NS * WV)) * ((M + 16 * MS - 1) / (16 * MS));
    hipLaunchKernelGGL((gemm_a4w4_wg_kernel<MS, NS, WV>), dim3(std::min(tiles, 256)), // hangy: tiles 如果超过256，就只留下256, persistent warps
                       dim3(64 * WV), 0, stream, A, B, AScale, BScale,
                       reinterpret_cast<__hip_bfloat16*>(out), M, N, K, scaleN_pad);
}

struct WgTile
{
    int mt, nt;
    void (*launch)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, void*,
                   int, int, int, int, hipStream_t);
};
// Workgroup tiles (MTILE x NTILE = 16*MSUB x 64*NSUB). Wider wave tiles (NSUB 6,
// 8) cut loads and LDS reads per MFMA and give the heuristic more ways to
// quantize N onto the CUs; MSUB*NSUB <= 64 keeps the accumulators in AGPRs.
constexpr WgTile kWgTiles[] = {
    {256, 256, launch_wg<16, 4>}, {192, 256, launch_wg<12, 4>}, {160, 256, launch_wg<10, 4>}, {128, 256, launch_wg<8, 4>},
    {96, 256, launch_wg<6, 4>},   {64, 256, launch_wg<4, 4>},
    {128, 384, launch_wg<8, 6>},  {96, 384, launch_wg<6, 6>},   {64, 384, launch_wg<4, 6>},
    {96, 512, launch_wg<6, 8>},   {64, 512, launch_wg<4, 8>},
};

void gemm_a4w4_mxfp4_launch(const uint8_t* A,
                            const uint8_t* B,
                            const uint8_t* AScale,
                            const uint8_t* BScale,
                            void*          out,
                            int            M,
                            int            N,
                            int            K,
                            int            Mpad,
                            int            scaleN_pad,
                            void*          stream)
{
    // Grid: one wave (64 threads) per (16*MSUB) x (16*NSUB) output tile.
    auto launch = [&](auto msub, auto nsub) {
        constexpr int MS = decltype(msub)::value;
        constexpr int NS = decltype(nsub)::value;
        constexpr int MTILE = 16 * MS;
        constexpr int NTILE = 16 * NS;
        dim3 block(64);
        dim3 grid((N + NTILE - 1) / NTILE, (Mpad + MTILE - 1) / MTILE);
        hipLaunchKernelGGL((gemm_a4w4_mfma_kernel<MS, NS>), grid, block, 0,
                           reinterpret_cast<hipStream_t>(stream), A, B, AScale,
                           BScale, reinterpret_cast<__hip_bfloat16*>(out), M, N,
                           K, Mpad, scaleN_pad);
    };

    // Workgroup-cooperative path (LDS-shared A, 4 waves/WG, persistent).
    // Tile choice: minimize ceil(tiles / NCU) * (MTILE + 64) * NTILE -- rounds
    // of one tile per CU, each costing the tile area plus a fixed per-tile
    // overhead (worth ~64 rows: prologue/epilogue, and the lower MFMA:load
    // ratio of short tiles).
    if (K % 1024 == 0)
    {
        constexpr int NCU_WG = 256;
        int best = -1;
        long best_cost = 0, best_tiles = 0;
        for (int i = 0; i < (int)(sizeof(kWgTiles) / sizeof(kWgTiles[0])); ++i)
        {  // hangy: for each config
            const int mt = kWgTiles[i].mt, nt = kWgTiles[i].nt;
            const long tiles = (long)((M + mt - 1) / mt) * ((N + nt - 1) / nt);
            const long cost  = (tiles + NCU_WG - 1) / NCU_WG * (long)(mt + 64) * nt;
            if (best < 0 || cost < best_cost)
            {
                best       = i;
                best_cost  = cost;
                best_tiles = tiles;
            }
        }
        // Too few tiles to occupy a quarter of the GPU: the single-wave kernel's
        // small tiles expose more parallelism.
        if (best_tiles >= NCU_WG / 4)
        {
            std::cout << "[gemm_a4w4_mxfp4_launch] best = " << best << std::endl;
            kWgTiles[best].launch(A, B, AScale, BScale, out, M, N, K, scaleN_pad,
                                  reinterpret_cast<hipStream_t>(stream));
            return;
        }
    }

    // Tile-shape heuristic: prefer the largest tile (best reuse) whose workgroup
    // count still fills the GPU (~>= 2 workgroups per CU); otherwise shrink the
    // tile to expose more parallelism. Thresholds measured on gfx950 (256 CUs).
    constexpr int NCU        = 256;
    constexpr int FILL       = 2 * NCU;  // target minimum workgroup count
    auto grid_wgs = [&](int mt, int nt) {
        return (long)((Mpad + mt - 1) / mt) * ((N + nt - 1) / nt);
    };
    using m8 = std::integral_constant<int, 8>;
    using m4 = std::integral_constant<int, 4>;
    using m2 = std::integral_constant<int, 2>;
    using m1 = std::integral_constant<int, 1>;

    // Larger per-wave tiles issue more independent MFMAs per K-step, which is
    // what keeps the matrix pipe busy (the real limiter -- occupancy is not).
    // HUGE 128x128 = 64 MFMAs/chunk (256 AGPR accumulators). Step down when a
    // tile that big can't fill the GPU, trading ILP for workgroup parallelism.
    if (grid_wgs(128, 128) >= FILL)
        launch(m8{}, m8{});        // HUGE 128x128: deepest MFMA ILP
    else if (grid_wgs(64, 128) >= FILL)
        launch(m4{}, m8{});        // TALL 64x128: 32-MFMA ILP, fills small-M/
                                   // large-N where 128x128 can't (e.g. M=1024)
    else if (grid_wgs(64, 64) >= FILL)
        launch(m4{}, m4{});        // BIG  64x64
    else if (grid_wgs(32, 32) >= FILL)
        launch(m2{}, m2{});        // MID  32x32
    else
        launch(m1{}, m2{});        // SMALL 16x32: tiny M, maximize occupancy
}
