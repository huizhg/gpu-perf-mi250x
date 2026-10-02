import sys, torch
from kernels.gemm import gemm
from bench.shapes import SHAPES

def check(M, N, K, tag, BM, BN, BK):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = gemm(a, b, BM, BN, BK).float()
    ref = (a @ b).float()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()   # worst error, relative to output size
    ok = rel < 1e-2
    print(f"{'PASS' if ok else 'FAIL'}  {tag:8s} {M:6d} {N:6d} {K:6d}  tiles {BM}x{BN}x{BK}  rel err {rel:.1e}")
    return ok

if __name__ == "__main__":
    results = []
    for BM, BN, BK in [(32, 32, 32), (128, 128, 32)]:
        results += [check(M, N, K, tag, BM, BN, BK) for M, N, K, tag in SHAPES]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)