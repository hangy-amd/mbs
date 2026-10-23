"""Minimal single-shape driver for the mbs A4W4 GEMM (for profiling/timing).

Usage: python bench_one.py M N K [iters]
Prints mbs latency (us) and TFLOPS; runs `iters` timed launches.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import aiter  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402

M, N, K = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 50
torch.set_default_device("cuda")

quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x = torch.randn((M, K), dtype=torch.bfloat16)
w = torch.randn((N, K), dtype=torch.bfloat16)
x, x_scales_shuffle = quant_func(x, shuffle=True)
w, w_scales_shuffle = quant_func(w, shuffle=True)
wshuffle = shuffle_weight(w, layout=(16, 16))

import mbs_gemm_a4w4  # noqa: E402


def call():
    return mbs_gemm_a4w4.gemm_a4w4(x, wshuffle, x_scales_shuffle, w_scales_shuffle)


# warmup
for _ in range(10):
    call()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    call()
end.record()
torch.cuda.synchronize()
us = start.elapsed_time(end) * 1e3 / iters
tflops = M * N * K * 2 / us / 1e6
print(f"mbs  M={M} N={N} K={K}  {us:8.2f} us  {tflops:8.1f} TFLOPS")
