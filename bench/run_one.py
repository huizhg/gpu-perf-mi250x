import argparse, torch
from kernels.gemm import gemm_v4

p = argparse.ArgumentParser()
p.add_argument("--impl", choices=["triton", "rocblas"])
p.add_argument("--M", type=int)
p.add_argument("--N", type=int)
p.add_argument("--K", type=int)

args = p.parse_args()

a = torch.randn(args.M, args.K, device="cuda", dtype=torch.float16)
b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)
fn = gemm_v4 if args.impl == "triton" else torch.matmul

for _ in range(10):           # warm-up, includes autotuning for Triton
    fn(a, b)
torch.cuda.synchronize()
for _ in range(50):           # the runs we care about
    fn(a, b)
torch.cuda.synchronize()
