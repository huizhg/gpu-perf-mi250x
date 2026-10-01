import csv, torch, triton

def time_ms(fn):
    """Median runtime of fn() in milliseconds."""
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median")

def gemm_stats(M, N, K, ms):
    flops = 2 * M * N * K                       # one multiply + one add per term
    bytes_ = 2 * (M * K + K * N + M * N)        # FP16 = 2 bytes; read A and B, write C once
    tflops = flops / ms / 1e9                   # FLOP per ms -> TFLOP per s
    tbs = bytes_ / ms / 1e9                     # bytes per ms -> TB per s
    ai = flops / bytes_                         # arithmetic intensity, FLOP per byte
    return tflops, tbs, ai

def versions():
    return f"torch {torch.__version__} hip {torch.version.hip} triton {triton.__version__}"

def run(impl_name, fn, shapes, out_csv):
    """Benchmark fn(a, b) on every shape and write one CSV row per shape."""
    rows = []
    for M, N, K, tag in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ms = time_ms(lambda: fn(a, b))
        tf, tbs, ai = gemm_stats(M, N, K, ms)
        rows.append(dict(impl=impl_name, M=M, N=N, K=K, tag=tag,
                         ms=round(ms, 5), tflops=round(tf, 2), tbs=round(tbs, 3), ai=round(ai, 1)))
        print(f"{impl_name:8s} {tag:8s} {M:6d} {N:6d} {K:6d}  {ms:8.4f} ms  {tf:7.2f} TFLOPS  {tbs:6.3f} TB/s")
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {versions()}\n")
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)