import torch
from bench.harness import time_ms, gemm_stats
from kernels.gemm import gemm_v2

BM, BN, BK, NW = 128, 256, 32, 8      # winner for 4096 ^ 3, 128, 256, 32, 8,8, from autotune_v3.log
for n in (4096, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for mi in (16, 32):
        for w in (0, 1, 2, 3):
            ms = time_ms(lambda: gemm_v2(a, b, BM, BN, BK, 8, NW,
                                         matrix_instr_nonkdim=mi, waves_per_eu=w))
            print(f"{n:5d}  mfma {mi}x{mi}  waves_per_eu={w}  {gemm_stats(n, n, n, ms)[0]:6.1f} TFLOPS")