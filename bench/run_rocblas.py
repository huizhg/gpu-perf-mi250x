from bench.harness import run
from bench.shapes import SHAPES

# def run(impl_name, fn, shapes, out_csv):
run ("rocBLAS", lambda a, b: a @ b, SHAPES, "results/rocblas.csv")

