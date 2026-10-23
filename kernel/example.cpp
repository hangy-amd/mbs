#include "example.h"
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <type_traits>
#include <utility>
#include <algorithm>
#include <string>
#include <unordered_map>

// Plain loop over raw buffers -- no torch, no ATen.
template <typename T>
void tensor_add(const T* a, const T* b, T* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
    std::cout << "hahah: in tensor_add" << std::endl;
}

// Explicit instantiations so the kernel library exports concrete symbols.
template void tensor_add<float>(const float*, const float*, float*, int64_t);
template void tensor_add<double>(const double*, const double*, double*, int64_t);

// __global__ void read_pattern_kernel1(const uint8_t* x, size_t x_stride, uint8_t* y) {
//     uint8_t buffer[64];
//     size_t offset = blockIdx.x * x_stride + threadIdx.x * 64;
//     #pragma unroll
//     for (int i = 0; i < 64; i++) buffer[i] = x[offset + i];
//     #pragma unroll
//     for (int i = 0; i < 64; i++) y[offset + i] = buffer[i];
// }

__global__ void read_pattern_kernel2(const uint8_t* x, size_t x_stride, uint8_t* y) {
    uint8_t buffer[64];
    size_t offset = blockIdx.x * x_stride + threadIdx.x * 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++)
            buffer[i * 16 + j] = x[offset + j];
        offset += 64 * 16;
    }
    offset = blockIdx.x * x_stride + threadIdx.x * 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++)
            y[offset + j] = buffer[i * 16 + j];
        offset += 64 * 16;
    }
}

#ifndef MBS_ASM_DIR
#define MBS_ASM_DIR "build/asm"
#endif

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t e_ = (cmd);                                                      \
        if (e_ != hipSuccess) {                                                     \
            std::cerr << #cmd << " failed: " << hipGetErrorString(e_) << std::endl; \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

// Load the asm build of a read_pattern kernel (build/asm/<kernel_name>.hsaco)
// once per name and cache the function handle.
static hipFunction_t get_read_pattern_asm_kernel(const std::string& kernel_name) {
    static std::unordered_map<std::string, hipFunction_t> cache;
    auto it = cache.find(kernel_name);
    if (it != cache.end()) return it->second;

    const std::string path = std::string(MBS_ASM_DIR) + "/" + kernel_name + ".hsaco";
    // Itanium mangling of `void <name>(const uint8_t*, size_t, uint8_t*)`.
    const std::string symbol =
        "_Z" + std::to_string(kernel_name.size()) + kernel_name + "PKhmPh";

    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, path.c_str()));
    hipFunction_t f;
    HIP_CHECK(hipModuleGetFunction(&f, mod, symbol.c_str()));
    cache.emplace(kernel_name, f);
    return f;
}

void read_pattern(const uint8_t* x, size_t x_stride, uint8_t* y) {
    std::cout << "x_stride = " << x_stride << std::endl;
    // for (int i = 0; i < 10; i++) read_pattern_kernel1<<<256, 64, 0, 0>>>(x, x_stride, y);
    // for (int i = 0; i < 10; i++) read_pattern_kernel2<<<256, 64, 0, 0>>>(x, x_stride, y);
    // Same signature as read_pattern_kernel1: (const uint8_t*, size_t, uint8_t*).
    void* args[] = {&x, &x_stride, &y};
    auto fn1 = get_read_pattern_asm_kernel("read_pattern_kernel1");
    auto fn2 = get_read_pattern_asm_kernel("read_pattern_kernel2");
    for (int i = 0; i < 10; i++)
        HIP_CHECK(hipModuleLaunchKernel(fn1,
                                        256, 1, 1,  // grid
                                        64, 1, 1,   // block
                                        0, 0, args, nullptr));
    for (int i = 0; i < 10; i++)
        HIP_CHECK(hipModuleLaunchKernel(fn2,
                                        256, 1, 1,  // grid
                                        64, 1, 1,   // block
                                        0, 0, args, nullptr));
}
