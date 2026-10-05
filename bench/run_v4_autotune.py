from bench.harness import run
from bench.shapes import SHAPES
from kernels.gemm import gemm_v4

run("triton_v4", gemm_v4, SHAPES, "results/triton_v4.csv")