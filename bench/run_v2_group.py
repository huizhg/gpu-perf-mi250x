import torch
from bench.harness import time_ms, gemm_stats
from kernels.gemm import gemm_v2

for n in (4096, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for gm in (1, 4, 8, 16):
        ms = time_ms(lambda: gemm_v2(a, b, 128, 128, 32, gm))
        print(f"{n:5d}  GROUP_M={gm:2d}  {gemm_stats(n, n, n, ms)[0]:6.1f} TFLOPS")