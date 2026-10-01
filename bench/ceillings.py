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
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
mm_runs = []
for _ in range(3):
    ms = time_ms(lambda: a @ b)
    mm_runs.append(2 * 8192**3 / ms / 1e9)             # TFLOPS

result = {
    "bandwidth_TBs": round(statistics.median(bw_runs), 3),
    "matmul_TFLOPS": round(statistics.median(mm_runs), 1),
    "bandwidth_runs": [round(v, 3) for v in bw_runs],
    "matmul_runs": [round(v, 1) for v in mm_runs],
    "versions": versions(),
}
print(json.dumps(result, indent=2))
json.dump(result, open("results/ceilings.json", "w"), indent=2)