// Layout-decode probe for the gfx950 scaled MXFP4 MFMA
//   v = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c,
//          fmt_a, fmt_b, opsel_a, scale_a, opsel_b, scale_b)
//
// Two layout-AGNOSTIC experiments (no assumption about lane<->(m,n,k) mapping):
//   1. scalar-scale check: A=B=all 1.0 (fp4 code 0x2), vary scale_a/scale_b/opsel
//      to learn whether the scale is one scalar over the whole operand or 4
//      per-32-K-block bytes packed in the 32-bit scale word.
//   2. unity sanity: A=B=1.0, scale=127 -> every C == K == 128.
//
// Build: make probe ; run: ./mfma_probe

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));

// fp4 e2m1 code 0x2 == +1.0; pack 8 nibbles per int32, two codes per byte.
__device__ __forceinline__ i32x8 broadcast_fp4(unsigned code)
{
    unsigned byte = (code & 0xF) | ((code & 0xF) << 4);
    unsigned w    = byte | (byte << 8) | (byte << 16) | (byte << 24);
    i32x8 v;
    for (int i = 0; i < 8; ++i)
        v[i] = static_cast<int>(w);
    return v;
}

// out layout: [case][64 lanes][4 acc]. opsel_a/opsel_b are compile-time.
template <int OPSEL_A, int OPSEL_B>
__global__ void probe(float* out, int scale_a, int scale_b, int case_id)
{
    int lane = threadIdx.x;
    i32x8 a  = broadcast_fp4(0x2);  // +1.0
    i32x8 b  = broadcast_fp4(0x2);  // +1.0
    f32x4 c  = {0.f, 0.f, 0.f, 0.f};

    // case_id 8: per-lane scale_a = 127 + lane/16 (group g in {0..3} -> 2^g).
    // If scale is per-lane VGPR and lane group g owns K-block g, every output
    // should be 32*(1+2+4+8) = 480.
    if (case_id == 8)
        scale_a = 127 + (lane >> 4);
    // case_id 9/10: decode C-output layout. row index of a fragment lane is
    // lane%16. Make A carry 2^row via scale_a, B carry 2^col via scale_b.
    //   case 9: C[m][n] = 128 * 2^m  (independent of n) -> decodes m.
    //   case 10:C[m][n] = 128 * 2^n  (independent of m) -> decodes n.
    if (case_id == 9)
        scale_a = 127 + (lane & 15);
    if (case_id == 10)
        scale_b = 127 + (lane & 15);

#if defined(__gfx950__)
    // fmt 4 = fp4 (e2m1) for both operands.
    f32x4 r = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 4, 4, OPSEL_A, scale_a, OPSEL_B, scale_b);
    for (int i = 0; i < 4; ++i)
        out[(case_id * 64 + lane) * 4 + i] = r[i];
#else
    for (int i = 0; i < 4; ++i)
        out[(case_id * 64 + lane) * 4 + i] = -1.0f;
#endif
}

template <int OPSEL_A, int OPSEL_B>
static void run(float* d_out, int scale_a, int scale_b, int case_id,
                const char* desc)
{
    hipLaunchKernelGGL((probe<OPSEL_A, OPSEL_B>), dim3(1), dim3(64), 0, 0, d_out,
                       scale_a, scale_b, case_id);
    hipDeviceSynchronize();
    float h[64 * 4];
    hipMemcpy(h, d_out + case_id * 64 * 4, sizeof(h), hipMemcpyDeviceToHost);
    // All lanes/acc are identical in these uniform experiments; print lane0 acc0
    // plus min/max across the tile to confirm uniformity.
    float v0 = h[0], mn = h[0], mx = h[0];
    for (int i = 0; i < 64 * 4; ++i)
    {
        mn = h[i] < mn ? h[i] : mn;
        mx = h[i] > mx ? h[i] : mx;
    }
    printf("%-52s C[0]=%12.6f  min=%12.6f  max=%12.6f\n", desc, v0, mn, mx);
}

int main()
{
    setbuf(stdout, nullptr);
    float* d_out = nullptr;
    hipMalloc(&d_out, sizeof(float) * 16 * 64 * 4);

    printf("MXFP4 16x16x128 scaled-MFMA probe (A=B=1.0, K=128)\n");
    printf("Expect unity case C==128. Decode scale encoding from the rest.\n\n");

    run<0, 0>(d_out, 127, 127, 0, "scale_a=127 scale_b=127 (unity)");
    run<0, 0>(d_out, 128, 127, 1, "scale_a=128 (2^1)  scale_b=127");
    run<0, 0>(d_out, 126, 127, 2, "scale_a=126 (2^-1) scale_b=127");
    run<0, 0>(d_out, 128, 128, 3, "scale_a=128 scale_b=128 (2^1 * 2^1)");
    // Packed 4-byte scale word: bytes {127,128,129,130} = 2^0,2^1,2^2,2^3.
    // If per-32-block: C = 32*(1+2+4+8)=480. If scalar(low byte): C=128*1=128.
    int packed = 127 | (128 << 8) | (129 << 16) | (130 << 24);
    run<0, 0>(d_out, packed, 127, 4, "scale_a=packed{127,128,129,130} opsel_a=0");
    run<1, 0>(d_out, packed, 127, 5, "scale_a=packed opsel_a=1 (byte1=128,2^1)");
    run<2, 0>(d_out, packed, 127, 6, "scale_a=packed opsel_a=2 (byte2=129,2^2)");
    run<3, 0>(d_out, packed, 127, 7, "scale_a=packed opsel_a=3 (byte3=130,2^3)");
    run<0, 0>(d_out, 0, 127, 8, "per-lane scale_a=127+lane/16 (expect 480)");

    // Decode C-output layout: (lane, acc) -> (m, n).
    hipLaunchKernelGGL((probe<0, 0>), dim3(1), dim3(64), 0, 0, d_out, 127, 127, 9);
    hipLaunchKernelGGL((probe<0, 0>), dim3(1), dim3(64), 0, 0, d_out, 127, 127, 10);
    hipDeviceSynchronize();
    float hm[64 * 4], hn[64 * 4];
    hipMemcpy(hm, d_out + 9 * 64 * 4, sizeof(hm), hipMemcpyDeviceToHost);
    hipMemcpy(hn, d_out + 10 * 64 * 4, sizeof(hn), hipMemcpyDeviceToHost);
    printf("\nC-output layout decode (lane, acc) -> (m, n):\n");
    for (int lane = 0; lane < 64; ++lane)
    {
        printf("  lane %2d:", lane);
        for (int i = 0; i < 4; ++i)
        {
            int m = (int)(log2f(hm[lane * 4 + i] / 128.0f) + 0.5f);
            int n = (int)(log2f(hn[lane * 4 + i] / 128.0f) + 0.5f);
            printf("  acc%d=(m%2d,n%2d)", i, m, n);
        }
        printf("\n");
    }

    hipFree(d_out);
    return 0;
}
