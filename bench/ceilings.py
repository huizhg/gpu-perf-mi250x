import json, statistics, torch
from bench.harness import time_ms, versions

# Bandwidth: copy 512 MB. Each element is read once and written once.
x = torch.empty(256 * 1024**2, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)
bw_runs = []
for _ in range(3):
    ms = time_ms(lambda: y.copy_(x))
    bw_runs.append(2 * x.numel() * 2 / ms / 1e9)      # (read + write) x 2 bytes -> TB/s

# Compute: a large square GEMM, where rocBLAS is at its best.
# Compute: best rocBLAS speed over several large shapes. An empirical roof, not a hardware limit.
COMPUTE_SHAPES = [(8192, 8192, 8192), (4096, 4096, 8192), (2048, 11008, 4096), (3840, 3840, 3840)]
by_shape = {}
for M, N, K in COMPUTE_SHAPES:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    runs = [2 * M * N * K / time_ms(lambda: a @ b) / 1e9 for _ in range(3)]
    by_shape[f"{M}x{N}x{K}"] = round(statistics.median(runs), 1)
mm_runs = list(by_shape.values())           # TFLOPS

result = {
    "bandwidth_TBs": round(statistics.median(bw_runs), 3),
    "matmul_TFLOPS": max(mm_runs),
    "bandwidth_runs": [round(v, 3) for v in bw_runs],
    "matmul_runs": [round(v, 1) for v in mm_runs],
    "versions": versions(),
    "matmul_by_shape": by_shape
}
print(json.dumps(result, indent=2))
json.dump(result, open("results/ceilings.json", "w"), indent=2)