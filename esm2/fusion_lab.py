"""Day 4 Fusion lab: ESM-2 3B FFN up-projection, y = GELU(x @ W^T + b), four contestants + your GEMM alone.

Run from the repo root:  run python -m esm2.fusion_lab            (M = 8192 tokens)
                         run python -m esm2.fusion_lab --M 16384  (optional second shape)
"""
import argparse, json, statistics, torch, torch.nn.functional as F
from bench.harness import time_ms
from kernels.gemm_gelu import gemm_v5_gelu, gemm_v5_plain

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=8192, help="tokens per batch")
args = p.parse_args()
M, K, N = args.M, 2560, 10240          # ESM-2 3B FFN: hidden 2560 -> intermediate 10240

# ---- Inputs, laid out exactly as the model holds them ----
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)            # activations (post-LayerNorm, ~unit scale)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02     # nn.Linear stores (out, in) = (10240, 2560)
b = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.02
wt = w.t()                                                            # (2560, 10240) view, no copy

# ---- The contestants ----
def floor():      return x @ wt                                # GEMM only: no bias, no GELU
def eager():      return F.gelu(F.linear(x, w, b))             # addmm (bias in the library GEMM) + separate GELU
_compiled = torch.compile(lambda x, w, b: F.gelu(F.linear(x, w, b)), dynamic=False)
def inductor():   return _compiled(x, w, b)                    # default mode: mm + fused bias-and-GELU kernel
def fused():      return gemm_v5_gelu(x, wt, b)                # your kernel, epilogue on
def gemm_alone(): return gemm_v5_plain(x, wt)                  # your kernel, epilogue off

contestants = [
    ("Floor", "x @ w.t() only", floor),
    ("Eager", "addmm + separate GELU", eager),
    ("Inductor default", "mm + fused bias-and-GELU kernel", inductor),
    ("Your fused kernel", "Triton GEMM + bias + exact GELU epilogue", fused),
    ("Your GEMM alone", "same kernel, epilogue off", gemm_alone),
]

# ---- 1. Correctness against an fp32 reference (also triggers autotuning and compilation, untimed) ----
with torch.no_grad():
    ref_gelu = F.gelu(F.linear(x.float(), w.float(), b.float()))
    ref_mm = x.float() @ wt.float()
    for name, _, fn in contestants:
        ref = ref_mm if name in ("Floor", "Your GEMM alone") else ref_gelu
        torch.testing.assert_close(fn().float(), ref, atol=2e-2, rtol=2e-2)
        print(f"ok  {name}")
    del ref_gelu, ref_mm

# ---- 2. Kernels per call, from the profiler (best effort; the timing below doesn't depend on it) ----
def kernels_per_call(fn, reps=3):
    try:
        from torch.profiler import profile, ProfilerActivity
        from torch.autograd import DeviceType
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
        return round(len(names) / reps), sorted(set(names))
    except Exception as e:
        print(f"  (kernel count skipped: {type(e).__name__})")
        return None, []

# ---- 3. Timing: time_ms (do_bench median) 5 times each, report the median and range ----
rows = []
for name, what, fn in contestants:
    n_k, k_names = kernels_per_call(fn)
    runs = [time_ms(fn) for _ in range(5)]
    rows.append(dict(name=name, what=what, kernels=n_k, kernel_names=k_names,
                     ms=statistics.median(runs), lo=min(runs), hi=max(runs)))
t = {r["name"]: r["ms"] for r in rows}
flop = 2 * M * N * K

print(f"\n### Fusion lab: {M} x {N} x {K}, bf16, exact GELU\n")
print("| Contestant | What runs | Kernels | Time (ms) | Range (ms) | vs floor | TFLOPS |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for r in rows:
    print(f"| {r['name']} | {r['what']} | {r['kernels'] if r['kernels'] is not None else '?'} "
          f"| {r['ms']:.3f} | {r['lo']:.3f} to {r['hi']:.3f} | {r['ms'] / t['Floor']:.2f}x "
          f"| {flop / r['ms'] / 1e9:.1f} |")

print("\nKernel names:")
for r in rows:
    print(f"  {r['name']}: " + ", ".join(n[:60] for n in r["kernel_names"]))

gemm_gap = t["Your GEMM alone"] - t["Floor"]
epilogue = t["Your fused kernel"] - t["Your GEMM alone"]
print(f"""
Saving vs Inductor (Inductor - fused): {t['Inductor default'] - t['Your fused kernel']:+.3f} ms   (predicted 0.3 to 0.5)
Your GEMM alone: {t['Your GEMM alone']:.3f} ms = {100 * t['Floor'] / t['Your GEMM alone']:.0f}% of the floor's speed
Why fused differs from the floor ({t['Your fused kernel'] - t['Floor']:+.3f} ms) = GEMM gap ({gemm_gap:+.3f}) + epilogue cost ({epilogue:+.3f})""")

out = f"results/fusion_lab_M{M}.json"
json.dump(dict(M=M, N=N, K=K, dtype="bf16", rows=rows), open(out, "w"), indent=2)
print(f"\nSaved {out}")