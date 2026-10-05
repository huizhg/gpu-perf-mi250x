from bench.harness import run
from bench.shapes import SHAPES
from kernels.gemm import gemm_v1_rows

CONFIGS = {"v1_rows": (128, 128, 32)}
for name, (BM, BN, BK) in CONFIGS.items():
    run(f"triton_{name}", lambda a, b: gemm_v1_rows(a, b, BM, BN, BK), SHAPES, f"results/triton_{name}.csv")