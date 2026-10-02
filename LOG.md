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
2. Compare each version's measured TFLOPS at 4096 with its prediction (20 and 80). If a version beats its prediction, some loads didn't come from HBM. Where could they come from? Hint: the MI250X has an 8 MB L2 cache per GCD.
3. Find a shape where v0 beats v1. (Look at decode.) Explain it by counting tiles against 110 CUs, as in the Day 2 lesson.

### End of day 3
- Tests: 36/36 passed
- v1 at 4096: 90.43 TFLOPS (89% of rocBLAS), v0: 19%
- Shape where v0 beats v1: decode shapes and the 512 512 square 
- Surprised me: I set the a_ptr += wrong.  wrote this in the beginning: a_ptrs += offs_k[None, : ] * s_ak
- Stuck on: stuck on understanding the pointer grids, how program is handing data. 
- First step tomorrow: read ROCm "Optimizing Triton kernels"