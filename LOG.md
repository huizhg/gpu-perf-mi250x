## Day 1 


### Versions
torch  2.10.0+rocm7.0  hip  7.0.51831
triton 3.6.0
GPUs   1 AMD Instinct MI250X

1. Occupancy: right idea, but two missing pieces

    You named one limit, shared memory (LDS on AMD). A small correction: LDS is allocated per workgroup (a block of wavefronts), not per wavefront. There are three limits, and whichever is tightest wins:

    Limit	MI250X	Charged per
    Registers (VGPRs)	512 per lane on each SIMD	wavefront
    LDS	64 KB per CU	workgroup
    Hardware wave slots	8 per SIMD	fixed cap

    For GEMM kernels on AMD, registers are usually the binding limit, not LDS. A worked example: a kernel using 128 VGPRs fits 512 / 128 = 4 waves per SIMD. If its workgroup of 4 waves also uses 32 KB of LDS, only 2 workgroups fit per CU. That's 8 waves across 4 SIMDs, so 2 per SIMD. LDS wins, and occupancy is 2.

    You also skipped why occupancy matters. When one wavefront waits hundreds of cycles for memory, the SIMD switches to another resident wavefront. More waves means more latency hidden. You'll compute exactly this for your own kernel on Day 4.

2. Tiling: right conclusion, slightly off mechanism

    It's not "back and forth". Each value makes one trip from HBM into LDS, then gets reused many times from there. The win is fewer total bytes read from HBM. Here's the arithmetic that makes it concrete. For a BM × BN tile, each step along K loads (BM + BN) × BK values and does 2 × BM × BN × BK FLOPs, so in FP16:

    intensity ≈ BM·BN / (BM + BN) FLOP/byte
    32 × 32 tile: 16 FLOP/byte
    128 × 128 tile: 64 FLOP/byte
    ridge point on MI250X: about 120 

    {
  "bandwidth_TBs": 1.257,
  "matmul_TFLOPS": 107.0,
  "bandwidth_runs": [
    1.26,
    1.257,
    1.257
  ],
  "matmul_runs": [
    107.4,
    107.0,
    106.8
  ],
  "versions": "torch 2.10.0+rocm7.0 hip 7.0.51831 triton 3.6.0"
}

measured ridge point: matmul TFlops / bandwidth (tb/s) = 107/1.257 = 85

    Geometrically, a bigger tile slides your kernel right along the roofline, out of the memory-bound region. That's your v0 vs v1 story for Day 3, already worked out.

3. Memory vs compute bound: correct, with one reversed cause and one missing case
    The causality is reversed. Low arithmetic intensity isn't caused by the bandwidth limit. It's a property of the algorithm (FLOPs per byte), and it's what makes the kernel bandwidth-bound. Place the kernel by its intensity first, then read off which roof caps it.
    The missing case: many kernels sit well below both roofs. They're latency-bound, from low occupancy, too few tiles to fill 110 CUs, or launch overhead. Your small and decode shapes will land there. The roofline tells you the gap exists; profiling tells you why.
    Vocabulary: on AMD, say "Matrix Cores" or "MFMA", not tensor cores. AMD interviewers notice.

### Triton tutorial 03
matmul-performance-fp16:
         M       N       K  rocBLAS (TFLOPS)  Triton (TFLOPS)
0    256.0   256.0   256.0          2.438549         2.383127
1    384.0   384.0   384.0          6.677253         6.740846
2    512.0   512.0   512.0         15.114610        13.421102
3    640.0   640.0   640.0         26.006348        22.598621
4    768.0   768.0   768.0         35.169629        29.187167
5    896.0   896.0   896.0         42.613930        40.320804
6   1024.0  1024.0  1024.0         46.928250        43.861108
7   1152.0  1152.0  1152.0         58.620546        53.679669
8   1280.0  1280.0  1280.0         68.265555        65.372572
9   1408.0  1408.0  1408.0         64.973858        55.736450
10  1536.0  1536.0  1536.0         73.775279        62.051654
11  1664.0  1664.0  1664.0         81.002152        70.492772
12  1792.0  1792.0  1792.0         83.062121        80.460639
13  1920.0  1920.0  1920.0         91.776407        66.923422
14  2048.0  2048.0  2048.0         85.967257        74.772456
15  2176.0  2176.0  2176.0         88.820851        81.979507
16  2304.0  2304.0  2304.0         91.217585        90.622810
17  2432.0  2432.0  2432.0         93.162087        88.224912
18  2560.0  2560.0  2560.0         96.954873        97.495756
19  2688.0  2688.0  2688.0         91.129503        80.493717
20  2816.0  2816.0  2816.0         97.460852        86.524476
21  2944.0  2944.0  2944.0         92.960569        92.933305
22  3072.0  3072.0  3072.0        100.998905        87.447884
23  3200.0  3200.0  3200.0         95.321347        92.313590
24  3328.0  3328.0  3328.0        110.277683        88.247442
25  3456.0  3456.0  3456.0        102.112067        93.169516
26  3584.0  3584.0  3584.0         94.716910        97.139170
27  3712.0  3712.0  3712.0         98.937819        94.047744
28  3840.0  3840.0  3840.0        111.452926        91.338413
29  3968.0  3968.0  3968.0        104.237234        95.939239
30  4096.0  4096.0  4096.0         98.972785        93.367901

### End of Day 1
Done

## Day 2
1, the four decode points sit on the left of the ridge point, they are memory-bound. 
2. I think the square matmul with shape 512 * 512, (the smallest tested shape) is furtheset below the roof, like 30% under the roof on the right of the ridge point. 
3. the small square shapes (512, 1024) sit under the roof but the big matrix (4096, 8192) sit near the roof. which is because the small matrix does not fill the 110 compute unit on the gpu, most of them are sitting idle for small shape matmul. 
4. there are two prefill matmul sit above the roof. shape :11008,4096. They are big matmul, I think it is because the ceiling measurement was too low, because what i set the ceiling shape is: 8192, 8192, which is smaller than the prefill shape 11008, 4096. 

(answer corrected by claude:
 1. Decode: correct, and there's more in it. They're memory-bound, yes. Now look closer at whether they sit on the slope. One M = 1 point reaches about 90% of the bandwidth line, but the other three sit at roughly 55%. A likely reason: a memory-bound kernel only pulls bandwidth through the CUs that are working. With 128-wide tiles, N = 4096 gives just 32 tiles for 110 CUs, so most CUs can't help fetch data. The usual fix is split-K: cut the K dimension into pieces so more workgroups run at once. That's exactly what the wvSplitK kernel you saw in your vLLM traces does. Check your CSV for which M = 1 shape is the 90% one; Day 5 will show which kernels rocBLAS picked.

2. Right shape, wrong size. 512 is the furthest below the roof, but it's about 7× below, not 30%. The point sits near 15 TFLOPS against a roof of 107. On log axes, each major gridline is 10×, so always read ratios, never distances. One more cause on top of your answer: 512³ takes only about 18 µs, and launching a kernel costs a few µs on its own, so fixed overhead eats a big share of the time.

3. Correct. In numbers: with 128 × 128 tiles, 512 gives 16 tiles and 1024 gives 64, both below 110 CUs.

4. Good catch, and you're half right about why. It isn't about size. rocBLAS's speed depends on how well a shape fits its tile choices and the 110 CUs, and 2048 × 11008 × 4096 simply fits better than 8192³. Your "compute roof" was never a hardware limit, just rocBLAS's best result on one shape. That's called an empirical roofline. Tomorrow's Step 0 fixes it: take the best over four large shapes, and add the datasheet roof as a dashed line. 

One fix in your log: 107 TFLOPS is 56% of the 191.5 datasheet peak, not 89%. And fill in "Surprised me" before you close the day. Your answer to question 4 is a perfect candidate.
)

### End of day 2: 
 ### End of day 2
- Ceilings: BW 1.257 TB/s (78% of datasheet), matmul 107 TFLOPS (89%), ridge 85 FLOP/B
- Furthest below the roof: 
- Surprised me:
- Stuck on:
- First step tomorrow: read MLC Data Layout + Tiled GEMM steps 1-3


## Day 3
 the compare table

square      512x   512x   512  rocBLAS   15.11   triton_v0    9.75 (  65%)   triton_v1    6.93 (  46%)
square     1024x  1024x  1024  rocBLAS   47.26   triton_v0   17.52 (  37%)   triton_v1   32.90 (  70%)
square     2048x  2048x  2048  rocBLAS   87.37   triton_v0   20.81 (  24%)   triton_v1   72.45 (  83%)
square     4096x  4096x  4096  rocBLAS  101.10   triton_v0   19.18 (  19%)   triton_v1   90.43 (  89%)
square     8192x  8192x  8192  rocBLAS  106.30   triton_v0   19.50 (  18%)   triton_v1   76.28 (  72%)
prefill    2048x  4096x  4096  rocBLAS  105.01   triton_v0   19.49 (  19%)   triton_v1   88.45 (  84%)
prefill    2048x 11008x  4096  rocBLAS  111.17   triton_v0   14.97 (  13%)   triton_v1   93.15 (  84%)
prefill    2048x  4096x 11008  rocBLAS  112.12   triton_v0   18.18 (  16%)   triton_v1   92.11 (  82%)
decode        1x  4096x  4096  rocBLAS    0.69   triton_v0    0.29 (  42%)   triton_v1    0.18 (  26%)
decode        4x  4096x  4096  rocBLAS    2.72   triton_v0    1.16 (  43%)   triton_v1    0.70 (  26%)
decode       16x  4096x  4096  rocBLAS   10.69   triton_v0    4.54 (  42%)   triton_v1    2.79 (  26%)
decode        1x 11008x  4096  rocBLAS    1.13   triton_v0    0.57 (  50%)   triton_v1    0.45 (  40%)
esm        1024x  3840x  1280  rocBLAS   76.35   triton_v0   20.83 (  27%)   triton_v1   62.73 (  82%)
esm        1024x  1280x  1280  rocBLAS   61.14   triton_v0   19.42 (  32%)   triton_v1   42.11 (  69%)
esm        1024x  5120x  1280  rocBLAS   83.47   triton_v0   21.03 (  25%)   triton_v1   76.26 (  91%)
esm        1024x  1280x  5120  rocBLAS   88.02   triton_v0   20.55 (  23%)   triton_v1   51.24 (  58%)
odd        1000x  1000x  1000  rocBLAS   39.18   triton_v0   13.97 (  36%)   triton_v1   18.38 (  47%)
odd        3000x  2000x  1500  rocBLAS   76.50   triton_v0   18.46 (  24%)   triton_v1   63.77 (  83%)

### End of Day 3
1. What is v1's speedup over v0 at 4096, and does the tile-intensity idea explain it?
v1 is 4.7× faster than v0 at 4096³. Tile intensity predicts 4×: v1's 128×128 tiles give 64 FLOP/byte, v0's 32×32 tiles give 16. Both kernels are bandwidth-bound, so 4× more reuse per byte gives roughly 4× more FLOPS.
2. Compare each version's measured TFLOPS at 4096 with its prediction (20 and 80). If a version beats its prediction, some loads didn't come from HBM. Where could they come from? Hint: the MI250X has an 8 MB L2 cache per GCD.
TFLOPS = bandwidth roof times intensity: 
v0 =  16 x 1.257 = 20; v1 = 64 x 1.257 = 80 . what measured at v0 for 4096 is around 20, 19.18 is just below the roof. but for v1, 90.43 is above the roof (80). 90.43 / 64 = 1.41 TB/s is above the bandwidth of HBM. which is a sing that v1 used memory that is faster than HBM. So some loads came from somewhere faster: the L2 cache. Programs that work on tiles in the same row all need the same strip of A. Programs in the same column all need the same strip of B. The first program fetches a strip from HBM. When a neighbour asks for the same strip soon after, it is still in the 8 MB L2, so the fetch never reaches HBM. This also explains the drop at 8192: the strips become too large to stay in L2 between reuses. 
but why v0 didn't break the hbm bandwidth? why it uses L2 less than v1? the tile size for v0 is 32, so there's 4096/32 =  128 tiles for a one row computation. 110 cu run 110 programs at onece. so the first wave cover most or row 1 All of those programs read the same strip of A. After the first one fetches it, the rest hit L2.

3. Find a shape where v0 beats v1. (Look at decode.) Explain it by counting tiles against 110 CUs, as in the Day 2 lesson.

v0 beats v1 on every decode shape and on 512³. At M = 1, N = 4096, v0's 32×32 tiles give 128 programs, enough to fill all 110 CUs. v1's 128×128 tiles give only 32, so 78 CUs sit idle. 1×11008 has the smallest gap because v1 gets 86 tiles there, close to filling the GPU. Big tiles raise intensity but cut the number of programs, so the best tile size depends on the shape.

### End of day 3
- Tests: 36/36 passed
- v1 at 4096: 90.43 TFLOPS (89% of rocBLAS), v0: 19%
- Shape where v0 beats v1: decode shapes and the 512 512 square 
- Surprised me: I set the a_ptr += wrong.  wrote this in the beginning: a_ptrs += offs_k[None, : ] * s_ak
- Stuck on: stuck on understanding the pointer grids, how program is handing data. 
- First step tomorrow: read ROCm "Optimizing Triton kernels"

## Day 4
result from bench.run_v2_gropu

4096  GROUP_M= 1    92.0 TFLOPS
 4096  GROUP_M= 4    92.1 TFLOPS
 4096  GROUP_M= 8    91.8 TFLOPS
 4096  GROUP_M=16    91.8 TFLOPS
 4096  GROUP_M=32    91.4 TFLOPS
 4096  GROUP_M=64    91.4 TFLOPS
 8192  GROUP_M= 1    98.0 TFLOPS
 8192  GROUP_M= 4    97.8 TFLOPS
 8192  GROUP_M= 8    89.5 TFLOPS
 8192  GROUP_M=16    81.4 TFLOPS
 8192  GROUP_M=32    77.5 TFLOPS
 8192  GROUP_M=64    76.6 TFLOPS

 data from bench.compare results/triton_v1.csv results/triton_v3.csv
square      512x   512x   512  rocBLAS   15.11   triton_v1    6.93 (  46%)   triton_v3   13.64 (  90%)
square     1024x  1024x  1024  rocBLAS   47.26   triton_v1   32.90 (  70%)   triton_v3   45.19 (  96%)
square     2048x  2048x  2048  rocBLAS   87.37   triton_v1   72.45 (  83%)   triton_v3   75.99 (  87%)
square     4096x  4096x  4096  rocBLAS  101.10   triton_v1   90.43 (  89%)   triton_v3   95.15 (  94%)
square     8192x  8192x  8192  rocBLAS  106.30   triton_v1   76.28 (  72%)   triton_v3  102.29 (  96%)
prefill    2048x  4096x  4096  rocBLAS  105.01   triton_v1   88.45 (  84%)   triton_v3   90.44 (  86%)
prefill    2048x 11008x  4096  rocBLAS  111.17   triton_v1   93.15 (  84%)   triton_v3   94.40 (  85%)
prefill    2048x  4096x 11008  rocBLAS  112.12   triton_v1   92.11 (  82%)   triton_v3   93.33 (  83%)
decode        1x  4096x  4096  rocBLAS    0.69   triton_v1    0.18 (  26%)   triton_v3    0.44 (  64%)
decode        4x  4096x  4096  rocBLAS    2.72   triton_v1    0.70 (  26%)   triton_v3    1.74 (  64%)
decode       16x  4096x  4096  rocBLAS   10.69   triton_v1    2.79 (  26%)   triton_v3    6.95 (  65%)
decode        1x 11008x  4096  rocBLAS    1.13   triton_v1    0.45 (  40%)   triton_v3    0.90 (  80%)
esm        1024x  3840x  1280  rocBLAS   76.35   triton_v1   62.73 (  82%)   triton_v3   68.31 (  89%)
esm        1024x  1280x  1280  rocBLAS   61.14   triton_v1   42.11 (  69%)   triton_v3   58.91 (  96%)
esm        1024x  5120x  1280  rocBLAS   83.47   triton_v1   76.26 (  91%)   triton_v3   80.35 (  96%)
esm        1024x  1280x  5120  rocBLAS   88.02   triton_v1   51.24 (  58%)   triton_v3   74.63 (  85%)
odd        1000x  1000x  1000  rocBLAS   39.18   triton_v1   18.38 (  47%)   triton_v3   26.54 (  68%)
odd        3000x  2000x  1500  rocBLAS   76.50   triton_v1   63.77 (  83%)   triton_v3   65.37 (  85%)

best-config table: 

| Family  | Shape                | BLOCK_M × BLOCK_N × BLOCK_K | num_warps | TFLOPS | % of rocBLAS |
|---------|----------------------|-----------------------------|-----------|--------|--------------|
| square  | 4096³                | 128 × 256 × 32              | 8         | 95.25  | 94%          |
| square  | 8192³                | 128 × 256 × 32              | 8         | 102.29 | 96%          |
| prefill | 2048 × 11008 × 4096  | 128 × 128 × 64              | 8         | 94.40  | 85%          |
| decode  | 1 × 4096 × 4096      | 64 × 64 × 64                | 4         | 0.44   | 64%          |
| esm     | 1024 × 1280 × 5120   | 128 × 128 × 64              | 8         | 74.63  | 85%          |

### look into the compiler
hzhang22@nid005028:/scratch/project_462001433/hui/gpu-perf> grep -h -E "NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy|LDSByteSize" $(find $TRITON_CACHE_DIR -name "*.amdgcn")
; NumVgprs: 107
; NumAgprs: 0
; TotalNumVgprs: 107
; ScratchSize: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; Occupancy: 4
hzhang22@nid005028:/scratch/project_462001433/hui/gpu-perf> grep -h -o '"shared": *[0-9]*' $(find $TRITON_CACHE_DIR -name "*.json")
"shared": 24576
TotalNumVgprs : Registers per lane for one wavefront (vector + accumulation registers together)

ScratchSize : Bytes spilled to slow memory because registers ran out. 0 is what you want

Occupancy: The compiler's answer: wavefronts per SIMD
"shared" (from the .json):  LDS bytes per workgroup. Use this if LDSByteSize shows 0

### compute occupancy
4c. Compute occupancy yourself, then check. On MI250X each SIMD has 512 registers per lane, allocated in chunks of 8, and holds at most 8 wavefronts. Each CU has 4 SIMDs and 64 KB of LDS.
1. Register limit: round TotalNumVgprs up to a multiple of 8, then 512 ÷ that, rounded down, capped at 8.
107 round up to 112. 512/112 = 4.5, round down to four. 

2. LDS limit: 65536 ÷ shared bytes, rounded down, gives workgroups per CU. Multiply by num_warps (wavefronts per workgroup), divide by 4 SIMDs: wavefronts per SIMD.
65536/24576, round donw = 2  2* num_warps = 8. 8/4 = 2 wavefrounds per workgroup

3. Occupancy is the smaller of the two.
occupancy is two? as 1 give us result 4, 2 give us result 2. the smaller number is 2.  but the occupancy from above is four. 

reason:The compiler sees zero bytes of LDS. Triton doesn't declare its shared memory statically in the kernel. It requests it as dynamic LDS at launch time, and that's where your 24576 comes from (kernel.metadata.shared). So when the compiler computed Occupancy: 4, the LDS limit looked infinite and only registers constrained it. It printed the register limit, which is exactly your step 1.
The true occupancy needs both constraints, and only you have both numbers:
At runtime, LDS binds first. The compiler's number is an upper bound, not the answer.

Two small fixes to your write-up:

In step 2, the unit at the end should be wavefronts per SIMD, not "per workgroup." The chain is: 2 workgroups/CU × 4 wavefronts/workgroup = 8 wavefronts/CU, then ÷ 4 SIMDs = 2 wavefronts/SIMD.
Step 3 can drop the question mark. Something like: "Occupancy = min(4, 2) = 2. The compiler reports 4 because it only sees static LDS (0 bytes); Triton's 24576 bytes are allocated dynamically at launch."

One consequence worth noting for later: registers are no longer your bottleneck. If you cut VGPRs from 107 to, say, 64, occupancy would still be 2. To raise it, you'd need to shrink shared memory, for example with smaller BLOCK_K or fewer pipeline stages (num_stages). Below 21845 bytes you get 3 workgroups per CU. Below 16384 bytes you get 4.
short: Compiler says 4, because it sees LDSByteSize: 0.
Hand calculation says 2, because the launch requests 24576 bytes.
Source of the 24576: kernel.metadata.shared.
Conclusion: actual occupancy is 2, bound by LDS, not registers.

### results from step 5 the knob table:
 4096  mfma 16x16  waves_per_eu=0    99.4 TFLOPS
 4096  mfma 16x16  waves_per_eu=1    99.2 TFLOPS
 4096  mfma 16x16  waves_per_eu=2    85.9 TFLOPS
 4096  mfma 16x16  waves_per_eu=3    86.1 TFLOPS
 4096  mfma 32x32  waves_per_eu=0    95.0 TFLOPS
 4096  mfma 32x32  waves_per_eu=1    95.0 TFLOPS
 4096  mfma 32x32  waves_per_eu=2    85.6 TFLOPS
 4096  mfma 32x32  waves_per_eu=3    85.9 TFLOPS
 8192  mfma 16x16  waves_per_eu=0   106.8 TFLOPS
 8192  mfma 16x16  waves_per_eu=1   106.7 TFLOPS
 8192  mfma 16x16  waves_per_eu=2    92.0 TFLOPS
 8192  mfma 16x16  waves_per_eu=3    92.1 TFLOPS
 8192  mfma 32x32  waves_per_eu=0   100.9 TFLOPS
 8192  mfma 32x32  waves_per_eu=1   100.8 TFLOPS
 8192  mfma 32x32  waves_per_eu=2    91.2 TFLOPS
 8192  mfma 32x32  waves_per_eu=3    91.2 TFLOPS

 1. 16 x 16 mfma is faster than 32 x 32. AMD's guide says 16 x 16 is faster on Mi300X, it also for Mi250x from our test. 
 2. the waves_per_eu did not help. with setting 0 has the bigest TFLOPS than 1,2,3. 

### discovery from day4



### End of Day 4. 
- GROUP_M at 8192: GROUP_M=1 _95__ TFLOPS, best _96.4__ (GROUP_M=_4_)
- v3 at 4096: _95.25__ TFLOPS (_94__% of rocBLAS); at 8192: 102.29, 96%of rocBLAS
- Occupancy by hand: _2_, compiler says: _4_
- MFMA 16 vs 32 on MI250X: _16__
- Surprised me: no
- Stuck on: computing the pid_m and pid_n with group_m. 
- First step tomorrow: read MLC appendix on measuring kernel performance

## day 5
### step 0a: launch order and the 8192 cliff

**Setup.** MI250X (one GCD), Triton, FP16, 128 × 128 × 32 tiles, median of `do_bench`.

### 1. A benchmarking trap: 82,000 TFLOPS

My first row-order kernel reported 0.0134 ms at 8192, or 82,000 TFLOPS and 30 TB/s.
One GCD peaks near 191.5 TFLOPS and 1.6 TB/s, so the number was impossible.
Cause: I dropped the final `tl.store` when copying the kernel. Nothing the kernel
computed was ever written, so the compiler deleted the K loop as dead code.
What remained was 4096 empty programs, about 13 µs of launch cost.
Fix: the harness now checks output against `a @ b` and asserts TFLOPS < peak
before writing any row.

### 2. Launch order explains the whole v1/v2 gap

v1 used a 2-D grid with `program_id(0)` as `pid_m`, so consecutive programs walk
down a column of output tiles. Swapping to row order:

| Shape | v1 (column walk) | v1_rows (row walk) |
|---|---|---|
| 512 to 2048 square | tie | tie |
| 4096 square | 90.4 | 91.6 |
| 8192 square | **76.3** | **98.2** |
| all prefill shapes | tie | tie |

Order only matters at 8192, and there it costs 22%.

### 3. GROUP_M reproduces it as a dose-response curve

In v2, larger GROUP_M makes the walk more column-like. GROUP_M = 1 is a pure row
walk; GROUP_M = num_pid_m is a pure column walk.

| GROUP_M | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4096 | 92.0 | 92.1 | 91.8 | 91.8 | 91.4 | 91.4 |
| 8192 | 98.0 | 97.8 | 89.5 | 81.4 | 77.5 | 76.6 |

- GROUP_M = 64 (76.6) matches v1 (76.3). GROUP_M = 1 (98.0) matches v1_rows (98.2).
  v1 and v2 are the same kernel; v2's index arithmetic costs nothing.
- At 8192, grouping adds nothing over a plain row walk. The common default of
  GROUP_M = 8 costs 9% here.
- The knee sits between GROUP_M = 4 and 8: 512 vs 1024 rows of A in flight.
- Run-to-run variation is about 3% (4096 moved from ~89.5 to ~92 between jobs).
  Only compare numbers from the same job.

### 4. Why: two hypotheses that failed

- **Symmetry.** I predicted row and column order would tie on a square problem.
  They don't.
- **Tile contiguity.** A tiles have 64-byte rows, B tiles 256-byte rows, so a column
  walk sends A's short fragments to HBM. But that would hurt at 4096 too, and it doesn't.

### 5. Why: what the data rules out

| Config | Rows of A in flight | Row stride | Span | Bytes per K-step | Loss |
|---|---|---|---|---|---|
| 4096, GROUP_M = 16 | 2048 | 8 KB | 16 MB | 128 KB | none |
| 4096, GROUP_M = 32 | 4096 | 8 KB | 32 MB | 256 KB | none |
| 8192, GROUP_M = 8 | 1024 | 16 KB | 16 MB | 64 KB | 9% |
| 8192, GROUP_M = 16 | 2048 | 16 KB | 32 MB | 128 KB | 17% |

- **Not L2 capacity.** GROUP_M = 16 moves the same 128 KB per K-step at both sizes,
  and only 8192 loses.
- **Not TLB reach.** 8192 with GROUP_M = 8 touches half the span and a quarter of the
  rows of 4096 with a full column walk, and still loses.
- **What's left is the stride.** At 8192, consecutive rows of A sit 16 KB apart.

### 6. The mechanism (working explanation)

In a column walk, the programs running at the same moment each own a different
block of rows of C. At every K-step they all read the same 32-column strip of A,
each from its own rows. Those reads land at addresses exactly 16 KB apart:
a large power of two.

The GPU decides which memory channel and which L2 cache set serve an address by
looking at some of its bits. Addresses that differ by a large power of two share
those bits, so they all go to the same few channels or sets. Those queue up or
evict each other while the rest of the memory system sits idle. More rows in
flight means more collisions, which is why the loss grows with GROUP_M.

A row walk avoids this. The programs in flight share the same rows of A, which
stay hot in L2, and they read a strip of B that is one contiguous block, which
spreads evenly across every channel.

At 4096 the stride is 8 KB, which evidently still spreads across enough channels
and sets. MI250X's exact address mapping isn't public, so where the threshold sits
has to be measured, not derived.

### 7. Still open

- [ ] **Padding test.** Pad each row by 64 elements (stride 16,512 bytes) and rerun
      GROUP_M = 64 at 8192. Recovery to ~98 confirms the stride as the cause.
- [ ] **rocprof counters** on padded vs unpadded. Lower L2 hit rate means set
      conflicts; normal hit rate with slower misses means channel camping.

### Lessons

1. Check every number against the hardware peak. A kernel that beats physics is broken.
2. Write the prediction down before running the test. Two of mine failed, and each
   failure narrowed the explanation.
3. Sweep the variable. One data point said "order matters"; the GROUP_M curve said
   *how much* and the 4096 vs 8192 comparison said *why*.
4. Tile order is a tunable parameter, not a default. At 8192, GROUP_M = 8 loses 9%.

###  with waves_per_eu = 0 (test day 4.b)
; NumVgprs: 107
; NumAgprs: 0
; TotalNumVgprs: 107
; ScratchSize: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; Occupancy: 4
hzhang22@nid005031:/scratch/project_462001433/hui/gpu-perf> grep -h -o '"shared": *[0-9]*' $(find $TRITON_CACHE_DIR -name "*.json")
"shared": 24576

with waves_per_eu = 2 (test day 4.b)

; NumVgprs: 107
; NumAgprs: 0
; TotalNumVgprs: 107
; ScratchSize: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; Occupancy: 2
hzhang22@nid005031:/scratch/project_462001433/hui/gpu-perf> grep -h -o '"shared": *[0-9]*' $(find $TRITON_CACHE_DIR -name "*.json")
"shared": 24576

### test results for 0b:
square      512x   512x   512  rocBLAS   15.11   triton_v3   13.64 (  90%)   triton_v4   14.10 (  93%)
square     1024x  1024x  1024  rocBLAS   47.26   triton_v3   45.19 (  96%)   triton_v4   44.44 (  94%)
square     2048x  2048x  2048  rocBLAS   87.37   triton_v3   75.99 (  87%)   triton_v4   76.80 (  88%)
square     4096x  4096x  4096  rocBLAS  101.10   triton_v3   95.15 (  94%)   triton_v4   97.98 (  97%)
square     8192x  8192x  8192  rocBLAS  106.30   triton_v3  102.29 (  96%)   triton_v4  105.26 (  99%)
prefill    2048x  4096x  4096  rocBLAS  105.01   triton_v3   90.44 (  86%)   triton_v4   92.09 (  88%)
prefill    2048x 11008x  4096  rocBLAS  111.17   triton_v3   94.40 (  85%)   triton_v4   96.25 (  87%)
prefill    2048x  4096x 11008  rocBLAS  112.12   triton_v3   93.33 (  83%)   triton_v4   94.40 (  84%)
decode        1x  4096x  4096  rocBLAS    0.69   triton_v3    0.44 (  64%)   triton_v4    0.46 (  67%)
decode        4x  4096x  4096  rocBLAS    2.72   triton_v3    1.74 (  64%)   triton_v4    1.82 (  67%)
decode       16x  4096x  4096  rocBLAS   10.69   triton_v3    6.95 (  65%)   triton_v4    7.23 (  68%)
decode        1x 11008x  4096  rocBLAS    1.13   triton_v3    0.90 (  80%)   triton_v4    0.97 (  86%)
esm        1024x  3840x  1280  rocBLAS   76.35   triton_v3   68.31 (  89%)   triton_v4   69.29 (  91%)
esm        1024x  1280x  1280  rocBLAS   61.14   triton_v3   58.91 (  96%)   triton_v4   55.19 (  90%)
esm        1024x  5120x  1280  rocBLAS   83.47   triton_v3   80.35 (  96%)   triton_v4   84.56 ( 101%)
esm        1024x  1280x  5120  rocBLAS   88.02   triton_v3   74.63 (  85%)   triton_v4   66.00 (  75%)
odd        1000x  1000x  1000  rocBLAS   39.18   triton_v3   26.54 (  68%)   triton_v4   27.23 (  69%)
odd        3000x  2000x  1500  rocBLAS   76.50   triton_v3   65.37 (  85%)   triton_v4   68.06 (  89%)


v4 (adds 16 × 16 MFMA to autotuning): 99% of rocBLAS at 8192, 97% at 4096. Prediction (~107 / ~99) held within 2%. Prefill 84 to 88%, likely wave quantization at M = 2048 (check winners). Decode 67 to 86%: too few programs (32 at N = 4096 vs 86 at N = 11008), the gap tracks program count. Two ESM regressions vs v3 to explain: config space or autotune noise.

### step 2 table 
| Run | Main kernel name | Calls | Average duration (µs) | Harness time (µs) | Harness vs kernel |
|---|---|---|---|---|---|
| rocblas_4096 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT256x128x32_MI32x32x1` | 60 | 1313.19 | 1359.38 | +3.5% |
| triton_4096 | `gemm_kernel_v2` | 1714 | 1350.21 | 1377.46 | +2.0% |
| rocblas_prefill | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT256x208x32_MI16x16x1` | 60 | 1618.73 | 1661.30 | +2.6% |
| triton_prefill | `gemm_kernel_v2` | 1350 | 1820.08 | 1847.07 | +1.5% |
| rocblas_decode | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT64x16x64_MI16x16x1` | 60 | 40.53 | 48.48 | +19.6% |
| triton_decode | `void at::native::vectorized_el...:array<char*, 1ul> >` | 6311 | 200.73 | 74.24 | -63.0% |

Notes:
- **rocblas_4096** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x32_MI32x32x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA0_LBSPPB128_LBSPPM0_LPA0_LPB8_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_2_MO1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_4096** full kernel name: `gemm_kernel_v2`
  - kernel_stats average is 1700.79 µs over all 1714 calls; the table uses the last 50 calls (1350.21 µs) because warm-up differs
  - also ran 1334x, 201.38 µs avg: `void at::native::vectorized_el...:array<char*, 1ul> >`
- **rocblas_prefill** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA2_GRVWB2_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA2048_LBSPPB128_LBSPPM0_LPA16_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_13_MO1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_prefill** full kernel name: `gemm_kernel_v2`
  - kernel_stats average is 2293.75 µs over all 1350 calls; the table uses the last 50 calls (1820.08 µs) because warm-up differs
  - also ran 1042x, 200.63 µs avg: `void at::native::vectorized_el...:array<char*, 1ul> >`
- **rocblas_decode** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA1024_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS2_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_decode** full kernel name: `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)`
  - also ran 7934x, 155.98 µs avg: `gemm_kernel_v2`

### answers for the questions under step 2 table:
1. Does the profiler agree with the harness? Yes for 4096 and prefill, within 1.5 to 3.5%, so your timing method is validated. The real exception is rocblas_decode, at +19.6%. That gap is 48.48 − 40.53 ≈ 8 µs per call, which is fixed launch overhead. It's invisible next to a 1300 µs kernel but large next to a 40 µs one. This is a finding, not an error. Triton decode will likely show the same thing after the fix.

2. Does rocBLAS switch kernels for decode? It stays in the same family but uses a very different configuration. Compare just the tile parts of the two names:

	Macro tile	MFMA
4096	MT256x128x32	MI32x32x1
decode	MT64x16x64	MI16x16x1

Both are Cijk_... Tensile MFMA GEMMs. There's no gemv, and SK0 means no stream-K. So rocBLAS does not switch to a dedicated GEMV kernel. It picks a much smaller tile from the same GEMM library. A quick sanity check on that kernel: decode reads the 4096 × 4096 FP16 weight matrix, about 33.5 MB. At 40.53 µs that is about 830 GB/s, roughly half of the 1.6 TB/s HBM peak. So even rocBLAS's decode kernel leaves bandwidth unused. That fits the "decode needs a different kind of kernel" finding.

3. What does the Triton kernel name look like? It's gemm_kernel_v2 in all three runs. That's expected, because gemm_v4 is the autotuner wrapped around gemm_kernel_v2 (see CONFIGS_V4 in Step 0c). The name does not include block sizes, so it can't confirm the config by itself. The autotune log (TRITON_PRINT_AUTOTUNING=1) is your evidence for which config ran.

### Three finds and the post outline:
1. the best tile depends on the shape. evidence: Autotuning took decode from 26% to 64% of rocBLAS, 512³ from 46% to 90%, ESM 1024 × 1280 × 5120 from 58% to 85% . And also for the use of group_m in kernel v2, we noticed that bigger group_m size will make the tile columns narrowers. the pros is that it make the data reused more, more l2 cahce hit than smaller group_m. but the bad thing is that bigger group uses less program, more CU sit idle, causing less TFlOPS. 

2. visiting order matters at 8192. Evidence: v1 at 76 vs row order at about 95 (confirmed or not by Step 0a); grouping added a little more. We need to consider the data layout while desigining a kernel. for row-wise data, moving across the column has less step size than move across the rows, which makes the loading faster i think. 

3. Decode needs a different kind of kernel. the matmul for decode process in LLM inference is a thin query multiply the previous KV cache. so the shape is something like 1x4096x4096. in the test, rocblas switched to another kernel for the decode shape. our autotuned handwritten kernel used the same kernel for different shape, which cause the performance gap between rocblas and ours.  

### End of day 5
- 0a launch order: v1 rows at 8192 = _98.2__ TFLOPS
- 0b waves_per_eu=2: VGPRs _107__ -> __107_, occupancy __4_ -> __2_, scratch _0_
- v4 at 4096: 97.98 (97% of rocBLAS); at 8192: 105.26 (99%)
- rocBLAS decode kernel: _Cijk decode	MT64x16x64	MI16x16x1
- Profiler vs harness agree within: _3.5_%
- Three findings: 1. visiting order matters at 8192  2. the best tile depends on the shape 3. decode needs a different kind of kernel
- Surprised me: no
- First step tomorrow: draft the post from the outline

## Day 6
Step 0 table: the unified headline numbers:
4096  v4       median   99.6  range   99.6 to   99.7 TFLOPS
4096  rocBLAS  median  101.5  range  101.4 to  101.9 TFLOPS
8192  v4       median  107.4  range  107.4 to  107.4 TFLOPS
8192  rocBLAS  median  106.5  range  106.5 to  106.5 TFLOPS