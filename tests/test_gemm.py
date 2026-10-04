import sys, torch
from kernels.gemm import gemm, gemm_v2, gemm_v3
from bench.shapes import SHAPES

from kernels.gemm import gemm, gemm_v2

def check(fn, name, M, N, K, tag):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out, ref = fn(a, b).float(), (a @ b).float()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    ok = rel < 1e-2
    print(f"{'PASS' if ok else 'FAIL'}  {name:10s} {tag:8s} {M:6d} {N:6d} {K:6d}  rel err {rel:.1e}")
    return ok

if __name__ == "__main__":
    versions = {
        "v1": lambda a, b: gemm(a, b, 128, 128, 32),
        "v2_g1": lambda a, b: gemm_v2(a, b, 128, 128, 32, 1),
        "v2_g8": lambda a, b: gemm_v2(a, b, 128, 128, 32, 8),
        "v3": lambda a,b: gemm_v3(a, b)
    }
    results = [check(fn, name, *s) for name, fn in versions.items() for s in SHAPES]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)