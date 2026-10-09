import statistics, torch
from bench.harness import time_ms, gemm_stats
from kernels.gemm import gemm_v4

for n in (4096, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    gemm_v4(a, b)                                   # autotune once, outside the timing
    for name, fn in (("v4", lambda: gemm_v4(a, b)), ("rocBLAS", lambda: a @ b)):
        runs = [gemm_stats(n, n, n, time_ms(fn))[0] for _ in range(5)]
        print(f"{n}  {name:8s} median {statistics.median(runs):6.1f}  range {min(runs):6.1f} to {max(runs):6.1f} TFLOPS")