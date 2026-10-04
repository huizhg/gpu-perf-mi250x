from bench.harness import run
from bench.shapes import SHAPES
from kernels.gemm import gemm_v3

run("triton_v3", gemm_v3, SHAPES, "results/triton_v3.csv")