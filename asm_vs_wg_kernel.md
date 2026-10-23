# gemm_a4w4_wg_kernel vs production asm f4gemm (gfx950)

Reference: `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
(disassembled with `llvm-objdump -d --mcpu=gfx950`), compared against
`gemm_a4w4_wg_kernel<16, 4, 4>` in `kernel/gemm_a4w4.hip`. Both use the same
256x256 workgroup tile.

## 1. What already matches

| Aspect | asm 256x256 | wg kernel |
|---|---|---|
| Workgroup | 256 threads, 4 waves; each wave has 64 accumulators (`a[0:255]`) = 256(M) x 64(N) | same (MSUB=16, NSUB=4, waves side by side along N) |
| Registers | 512 (256 VGPR + 256 AGPR), 1 wave/SIMD, 160 KB LDS | `__launch_bounds__(256,1)`, accumulators pinned to AGPRs via inline asm `"+a"` |
| B (weights) | `buffer_load_dwordx4` straight into VGPRs, ping-pong sets `v[136:167]` / `v[168:199]`, 1 stage ahead | same (`PF_B = 1` for this tile) |
| A (activations) | `buffer_load_dwordx4 ... lds` DMA, 8 ops/wave/stage (32 KB/WG), then `ds_read_b128` | same (8 DMA ops/wave/stage) |
| Operand order | B is the first MFMA source, so it computes C^T | same |
| Scales | one dword covers 4 MFMAs; byte picked with `op_sel` / `op_sel_hi` | same (`mfma_fp4_sel<OA,OB>`) |
| Stage | K=256 = 128 MFMAs/wave | same |
| Loads | spread one at a time between MFMAs, never issued in bursts | same (`issue_one` at `k*T_LOAD/L`) |
| Waits | hand-placed `s_waitcnt vmcnt(N)` | same (`W_START` / `WAIT_KEEP`) |

## 2. Implementation differences

### 2.1 Barriers and the shape of the K loop (biggest difference)
- **asm**: each K=256 stage is split into **two sections of 64 MFMAs**. Each
  section starts with `s_waitcnt vmcnt(10|15) lgkmcnt(0); s_barrier`.
  - Section 1 covers M rows 0-127 (`a[0:127]`) for both K-128 halves.
  - Section 2 covers M rows 128-255 (`a[128:255]`) for both K-128 halves.
  - Global loads are split by section. Section 1 issues the next stage's B
    (8 x dwordx4) and B scales (2 dwords) into VGPRs. Section 2 issues the
    A DMA (8 x dwordx4 lds) and A scales into LDS (2 x dword lds).
  - The loop body is unrolled 2 stages, with the B register sets swapped each stage.
- **wg**: one flat stream of `T = 128` MFMAs per stage, ordered
  `t -> (h, mi, ni)` (K half first, then all 16 M fragments, `ni` fastest), with
  **one** wait+barrier at `t = T_LOAD`.

### 2.2 How far ahead A fragments are read from LDS
- **asm**: reads in bulk, a whole section ahead. During section 1 it reads
  all 16 A fragments for section 2 into `v[72:135]`. During section 2 it reads
  the next stage's section-1 fragments into `v[8:71]`. That is 128 VGPRs of
  double-buffered A fragments, with about 64 MFMAs of latency slack. The four
  A-scale dwords per section (`ds_read_b32`) are read the same way.
- **wg**: `FRAG_AHEAD = 2` (3 for MSUB < 16) fragments ahead, about 8 MFMAs of
  slack. Only the `head[]` fragments cross the stage boundary.

### 2.3 MFMA order
- **asm**: interleaves both operands in a 2x2 zig-zag:
  (b0,a0), (b0,a1), (b1,a0), (b1,a1), (b0,a2), ... Neither operand is reused
  by back-to-back MFMAs for long, and each section covers 4 B x 8 A fragments
  per K half.
- **wg**: one A fragment is held for NSUB=4 consecutive MFMAs (`ni` fastest).

### 2.4 A scales
- **asm**: loaded **once per workgroup into LDS** (2 `buffer_load_dword ... lds`
  per wave per stage), then each wave reads them with `ds_read_b32`.
  20 global ops per wave per stage.
- **wg**: every wave loads all `MSUB/2 = 8` A-scale dwords from global memory
  into VGPRs, which is 4x redundant. 26 global ops per wave per stage.

### 2.5 LDS layout for A
- **asm**: no XOR swizzle. DMA destinations (`m0`) step by `0x1080`
  (4 KB + 128 B): the waves' 1 KB writes are interleaved, with 128 B of
  padding per 4 KB to avoid bank conflicts on the reads.
- **wg**: XOR swizzle (`chunk ^ (row & 7)`), with each wave's slice stored contiguously.

### 2.6 Epilogue
- **asm**: no LDS. For each pair of N fragments it does
  `v_accvgpr_read` -> `v_mul_f32` (alpha, `s38`) -> `v_cvt_pk_bf16_f32` ->
  `v_permlane16_swap_b32` (x2) -> `buffer_store_dwordx4`. 32 x 16 B stores per wave.
- **wg**: stages through LDS (`sepi`, ~18 KB for this tile): `ds_write_b64`,
  then `ds_read_b128`, then `buffer_store_b128`, with `sched_barrier` fencing per 32 rows.

### 2.7 Scheduling across tiles
- **asm**: not persistent. Each workgroup runs one tile, then the epilogue,
  then `s_endpgm`.
- **wg**: persistent (grid = min(tiles, 256)), uses an XCD-aware, GROUP_M=4 tile
  order, and keeps the software pipeline running across tile boundaries. This
  was the wg kernel's own idea, not taken from asm.

### 2.8 Hazards
- **asm**: explicit `s_nop` (210 of them in the kernel).
- **wg**: relies on the MFMA spacing being large enough (inline asm is opaque to
  the hazard recognizer).

### 2.9 Features only the asm kernel has
- A split-K path (`buffer_atomic_pk_add_bf16` epilogue).
- Alpha scaling of the output (`v_mul_f32` by `s38`).

## 3. Proposed further optimizations (in priority order) -- see §6 for results

Notes say the remaining gap to asm is per-MFMA overhead, not memory
traffic, so items 1-3 come first.

1. **Adopt the asm loop shape: 2 sections and 2 barriers per stage.**
   Split each stage into two M halves of 64 MFMAs. Issue the B and B-scale
   loads in section 1 and the A DMA and A-scale DMA in section 2. Place a
   `vmcnt + lgkmcnt(0) + s_barrier` at each section start.
2. **Bulk fragment reads, one section ahead.** Double-buffer the A fragments in
   VGPRs (2 x 16 x i32x4 = 128 VGPRs). While section k runs, `ds_read_b128`
   all fragments of section k+1, interleaved about one per two MFMAs. This
   replaces `FRAG_AHEAD` and `head[]`. Check the VGPR budget: B sets take 64,
   A fragments 128, scales about 16, addresses the rest of the 256.
3. **2x2 zig-zag MFMA order** inside a section, as in asm, instead of `ni`-fastest.
4. **Share A scales through LDS.** Each wave DMAs `MSUB/2/WAVES` scale dwords
   (`buffer_load_dword ... lds`) and reads the needed ones with `ds_read_b32`.
   This cuts 26 to 20 global ops per wave per stage, and frees VGPRs for item 2.
5. **Epilogue with `v_permlane16_swap_b32` instead of LDS staging.** Combine the
   accumulators of N fragments `ni` and `ni+1`, swap halves between lane groups,
   and issue `buffer_store_dwordx4` directly. This drops the ~18 KB `sepi` buffer
   (more LDS for the ring) and the ds_write/ds_read round trip. It must still
   issue a fixed number of stores per tile for the vmcnt bookkeeping
   (`EPI_STORES`).
6. **Consider the asm LDS layout** (padding of 0x1080 per 4 KB, interleaved waves)
   if bank conflicts show up after item 2 changes the read pattern.
7. **Keep persistence and the XCD-aware order.** The asm kernel lacks them.
   Re-measure whether they still help once the loop matches asm.

## 4. Correctness items to verify

- **Bounds checking with `soffset`.** Rows past M are handled entirely through
  `soffset`: the A DMA (`to.a + j*8*Kp`), the scales (`tob.as`), and the
  epilogue (`sbase = (m0 + mp*32)*N*2`). The kernel's comment says the hardware
  bounds-checks voffset + soffset. LLVM's intrinsic documentation instead
  describes soffset as "excluded from bounds checking". If that is true, then
  when M % 32 != 0 the epilogue writes past `out`, and the A DMA reads past A.
  The earlier "B base in soffset faults" observation fits this. Test with
  M = 1000 and canary memory after the output.
- **XCD assumption for small grids.** The XCD remap assumes `gridDim.x % 8 == 0`.
  When tiles < 256 and not a multiple of 8, that doesn't hold. Results stay
  correct, but L2 locality suffers.
- **Stale comments.**
  - `RING = PF+1` is used to justify refill safety, but the code has RING = 4, PF = 2.
  - The `mfma_fp4_acc` comment mentions "16 accumulators" and "non-volatile".
  - The single-wave `launch_bounds` comment mentions "occupancy 3".

## 5. Generated-code check (2026-10-19)

Compiled with the Makefile flags (`hipcc -O3 --offload-arch=gfx950 --cuda-device-only -S`).

**What matches the design, in every wg kernel build**
- The K loop is fully unrolled over RING=4 steps: 4 x 128 MFMAs, with no
  spills and no `v_accvgpr_*` copies anywhere between the first and last MFMA.
- The per-step stream is exactly as designed:
  `W_START`, then `[ds_read, lgkmcnt(2), 4 MFMAs]` x 30 with the 26 global ops
  threaded through (B, then scales, then A DMA), then `WAIT_KEEP + s_barrier`
  at T_LOAD=120, then 2 head `ds_read`s under the last 8 MFMAs.
- The wait immediates equal the constexpr formulas:
  - 256x256: `W_START=8`, `W_START_T=40`, `WAIT_KEEP = vmcnt(26) lgkmcnt(0)`.
  - 128x256: `22`, `38`, `18`.
- The compiler adds only `lgkmcnt(2)` before MFMAs (matching `FRAG_AHEAD=2`)
  and no vmcnt waits of its own.
- MFMAs use AGPR accumulators, B as src0, and op_sel scale bytes as designed.
  Every A DMA is preceded by `s_mov_b32 m0`.

**Deviation: the 256x256 (MSUB=16, NSUB=4) build spills**
- `vgpr_spill_count: 4`, `private_segment_fixed_size: 20`. This is the only
  wg build that spills.
- All the spill traffic is in the epilogue (`.LBB0_17`):
  - `scratch_load` (reload of the opaque-lane `v_mov_b32` input) is followed by
    `s_waitcnt vmcnt(0)` as the first thing in the epilogue.
  - 3 more `vmcnt(0)`/`vmcnt(1)` follow for the other reloads, including
    `a254/a255` spilled to scratch around a 74-instruction `v_accvgpr_mov` shuffle.
- `vmcnt(0)` drains the next tile's in-flight prefetch (A stages 0-1, B stage 0)
  and the epilogue stores already issued. So for the main large-shape tile,
  the cross-tile pipeline overlap is lost at every tile boundary. This is
  exactly what the opaque-lane trick was meant to avoid.
- Other builds with epilogue-only `v_accvgpr_mov` shuffles but no spills:
  12x4 (54), 6x8 (46), 8x6 (30), 10x4 (22).

**Fix ideas**
- Derive the epilogue lane id inside the asm (`v_mbcnt_lo/hi`) instead of
  keeping `threadIdx.x` live across the K loop.
- Read accumulators in their allocated order, to avoid the `accvgpr_mov`
  permutation and the a254/a255 spill.
- Or do the `permlane16_swap` epilogue (section 3, item 5), which needs fewer VGPRs.

## 6. Optimization round (2026-10-19, GPU 7, MI355X)

Timing: CUDA-graph replay of 100 launches, L2 warm, median of 7. "per stage"
comes from a regression of time against K at a single round of 256 tiles
(4096x4096). "cycles" come from the `MBS_TIMING=1` instrumentation (see
below).

**Result.** asm/mbs speed (geomean over the 14 model shapes) went from 0.895 to
0.927. The 256x256 per-stage cost went from 1902 to 1829 ns (asm: 1735). All
21 correctness shapes pass, including ragged M/N (M=1000, 777, 33, 300, 8000,
N=48/1040). `test_gemm_a4w4.py` reports err=0 on every shape.

| shape | mbs before (us) | mbs after (us) | asm (us) |
|---|---|---|---|
| 8192x17408x5120 | 413 | 396 | 369 |
| 8192x5120x17408 | 378 | 364 | 342 |
| 8192x10240x5120 | 232 | 221 | 211 |
| 8192x12288x5120 | 279 | 268 | 252 |
| 1024x17408x5120 | 59.2 | 58.0 | 55.9 |
| 1024x1024x5120 | 13.5 | 14.0 | 7.3 (only 64 tiles, see open items) |

**What changed in the kernel (in order of measured effect)**
1. **Epilogue writes whole 128 B row segments.** `permlane16_swap` then DPP
   `row_ror:8` + select, no LDS. Each store now covers 8 rows x 128 B instead
   of 16 rows x 64 B. A microbenchmark with all CUs storing at once:
   half-line store pairs take ~7.5k cycles until acked, full lines ~3.0k.
   That matters because vmcnt retires in order: the next tile's loads queue
   behind the stores. Done together with the next item: about +2% geomean.
2. **No more tile-boundary spills (256x256).** Each accumulator read is pinned
   with `asm volatile("" : "=v"(x) : "0"(acc))`. Without it the register
   allocator hoists all 256 AGPR reads to the top of the epilogue. The reset
   is `acc = 0; asm volatile("" : "+a"(acc))`; a plain `= 0` gets folded into
   the tile-loop phi and causes the accumulator shuffle + spill seen in §5.
   The 256-row tile also passes its scale loads' block offset via SGPR
   soffset (4 VGPRs saved). 0 VGPR spills now, and no `vmcnt(0)` outside the
   kernel end.
3. **Asm-style load addressing (§2: "loads").** Each load has a fixed VGPR
   offset, soffset is 0, and the per-stage/per-tile part goes into the
   descriptor base (range shrunk to match). Measured: an SGPR soffset makes
   the SQ spend 2 cycles to send each VMEM instruction instead of 1
   (`SQ_INST_CYCLES_VMEM_RD`, asm: exactly 1.0/inst). The SALU chain the
   compiler emitted in front of every load also stalled the stream. Putting
   the offset in voffset with a v_add per load was *slower* (+10% section-0
   body). Per stage: 3011 -> 2833 cycles.
4. **Loop shape = items 1-3 of §3.** Two sections per stage, A fragments
   (and A scales) read a whole section ahead, 2x2 zig-zag MFMA order. Register
   loads go in section 0 and the A DMA in section 1. Each section starts
   with `lgkmcnt(0)`; vmcnt + barrier only at section 1. On its own this was
   performance-neutral, but every later item builds on it.
5. **A scales shared through LDS (§3 item 4).** One DMA per scale block per
   workgroup, then `ds_read_b32`: 20 instead of 26 global ops per wave per
   stage. Neutral alone.
6. **Section 0 without a barrier** (not needed with RING = 4 > PF+1): -60
   cycles/stage, ~1%. Loads in the first 7/8 of each section instead of
   spread over all of it: ~-2% per stage in the timing harness, <1% end to
   end.

**What was measured and dropped (all within +-2%)**
- `FRAG_AHEAD` 2 -> 4 (old loop): LDS read latency was not the limiter. No
  LDS bank conflicts in either kernel (`SQ_LDS_BANK_CONFLICT` = 0).
- Non-persistent launch, GROUP_M 2/8/16, staggered workgroup start.
- Issuing the first 1/2/4 MFMAs of section 1 before its barrier, and a
  read-ahead window for the fragment reads.
- `s_mov m0` before every DMA is not what makes DMAs costly (same m0 for all
  DMAs measured identical).

**Where the time goes now (256x256, cycles per wave per stage, 4096^2 x 16384)**
- No loads at all: 2290 (16.7 cycles/MFMA, the MFMA floor).
- Now: ~2710, of which section-0 body (B loads) +170 and section-1 body
  (A DMA) +230 over that floor, waits ~180.
- The asm is ~2570 at the same clock. So the remaining per-stage gap (~5%) is
  in how the global loads interleave with the MFMAs, not in waits.
- On multi-round shapes the epilogue is ~15% of the 256x256 kernel. That is
  mostly store issue (~68 cycles per 1 KB store per CU, store data path) and
  is shared with the asm kernel, which also waits for its stores before
  `s_endpgm`.

**Open items**
- 1024x1024x5120 (0.53x): 64 tiles fill a quarter of the GPU. The asm uses
  32x128 (256 tiles). A 32-row wg tile does not fit the two-section loop
  (MS1 = 0). Split-K (the asm's `buffer_atomic_pk_add_bf16` path, §2.9) is
  the natural fix.
- M=1024 shapes with 96/160-row tiles are still 4-8% behind. Their per-stage
  cost was not analysed separately.
- `v_cndmask_b32_dpp` could fuse the epilogue's DPP mov + select (128 VALU
  per tile).

**§4 correctness items, resolved**
- Bounds checking: verified with a microtest on gfx950. The buffer range
  check covers voffset + soffset + the instruction offset. A load/store with
  voffset 60 + soffset 8 against a 64 B descriptor is dropped, and so is
  voffset 32 + `offset:32`. Rows past M are therefore clipped in hardware,
  and the descriptor-base addressing of item 3 keeps that property.
- The stale comments (RING = PF+1, "16 accumulators" / "non-volatile",
  "occupancy 3") are fixed.

**Tools left in the source**
- `-DMBS_TIMING=1` adds per-phase `s_memtime` buckets (section waits/bodies,
  epilogue, post-epilogue waits). Read them with `mbs_timing_read()` from
  `libmbs_gemm_kernel.so` (ctypes). Note that its extra SGPRs perturb the
  256x256 build.
- `-DMBS_FORCE_TILE_ENV` lets `MBS_TILE=<kWgTiles index>` force a wg tile.
- The thread-trace decoder (`rocprof-trace-decoder`) is not installed in this
  container, so `rocprofv3 --att` fails. The TA_* counter group hung
  rocprofv3 on the asm kernel.
