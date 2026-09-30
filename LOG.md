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