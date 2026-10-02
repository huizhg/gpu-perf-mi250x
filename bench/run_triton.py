from bench.harness import run
from bench.shapes import SHAPES
from kernels.gemm import gemm

CONFIGS = {"v0": (32, 32, 32), "v1": (128, 128, 32)}
for name, (BM, BN, BK) in CONFIGS.items():
    run(f"triton_{name}", lambda a, b: gemm(a, b, BM, BN, BK), SHAPES, f"results/triton_{name}.csv")