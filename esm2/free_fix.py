"""Post 2, Step 0: is there a free fix?

Can Inductor match your hand-fused GEMM + bias + GELU kernel if it is allowed to write the GEMM itself?
Same shape and inputs as esm2/fusion_lab.py, and your kernel is re-timed in the same session.

Run from the repo root:
    run python -m esm2.free_fix 2>&1 | tee results/free_fix.log
Prefix with TORCH_LOGS=output_code to also print the Triton source Inductor generated.
"""
import argparse, json, statistics, time
import torch, torch.nn.functional as F
from bench.harness import time_ms
from kernels.gemm_gelu import gemm_v5_gelu

DEVICE = "cuda"
MAX_AUTOTUNE = {"max_autotune": True, "coordinate_descent_tuning": True}   # = mode "max-autotune-no-cudagraphs"

# Compiled contestants. Each needs its own lambda: Dynamo caches compiled code per function,
# so compiling one function three ways could silently reuse the first compilation.
COMPILED = [
    ("Inductor default",
     "library GEMM + fused bias-and-GELU kernel",
     lambda x, w, b: F.gelu(F.linear(x, w, b)),
     dict(dynamic=False)),
    ("Inductor max-autotune",
     "Inductor picks: library GEMM or its own Triton GEMM",
     lambda x, w, b: F.gelu(F.linear(x, w, b)),
     dict(dynamic=False, options=dict(MAX_AUTOTUNE))),
    ("Inductor max-autotune, Triton only",
     "Inductor must write the GEMM, so it can fuse the epilogue",
     lambda x, w, b: F.gelu(F.linear(x, w, b)),
     dict(dynamic=False, options=dict(MAX_AUTOTUNE, max_autotune_gemm_backends="TRITON"))),
]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def kernels_per_call(fn, reps=3):
    """GPU kernels launched per call, from the profiler: what actually ran, whatever the logs say."""
    try:
        from torch.profiler import profile, ProfilerActivity
        from torch.autograd import DeviceType
        fn(); sync()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            sync()
        names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
        if not names:
            return None, []
        return round(len(names) / reps), sorted(set(names))
    except Exception as e:
        print(f"  (kernel count skipped: {type(e).__name__})")
        return None, []


def describe(count, names):
    """Turn kernel names into a verdict. triton_tem_ = a GEMM Inductor wrote; Cijk_ = the library (Tensile)."""
    if not names:
        return "unknown (no kernel names)"
    tem = [n for n in names if n.startswith("triton_tem_")]
    lib = [n for n in names if n.startswith("Cijk")]
    if tem and count == 1:
        return "FUSED: Inductor's own GEMM with bias + GELU in its epilogue"
    if tem:
        return "Inductor's own GEMM, but GELU still runs as a separate kernel"
    if lib:
        return "library GEMM" + (" + separate elementwise kernel" if (count or 1) > 1 else "")
    return "other: " + ", ".join(names)


def decide(fused_ms, compiled):
    """compiled: {name: ms} for compiled contestants that passed correctness."""
    if not compiled:
        return "No compiled contestant passed. Note it as an open question for the post."
    name, ms = min(compiled.items(), key=lambda kv: kv[1])
    gap = (ms - fused_ms) / fused_ms
    if gap <= 0:
        return (f"Recommend the compiler: '{name}' runs at {ms:.3f} ms, faster than your kernel's "
                f"{fused_ms:.3f} ms ({100 * gap:+.1f}%), with nothing extra to maintain.")
    if gap <= 0.02:
        return (f"Recommend the compiler: '{name}' runs at {ms:.3f} ms vs your kernel's {fused_ms:.3f} ms "
                f"({100 * gap:+.1f}%). Same speed, nothing extra to maintain.")
    return (f"Your kernel earns its place: {ms - fused_ms:.3f} ms ({100 * gap:.1f}%) faster than the best "
            f"compiler setting, '{name}' at {ms:.3f} ms.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192, help="tokens per batch")
    p.add_argument("--K", type=int, default=2560, help="hidden size")
    p.add_argument("--N", type=int, default=10240, help="FFN intermediate size")
    args = p.parse_args()
    M, K, N = args.M, args.K, args.N

    # Inputs exactly as in fusion_lab.py: same seed, nn.Linear layout, small weights like a trained model
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
    b = torch.randn(N, device=DEVICE, dtype=torch.bfloat16) * 0.02
    wt = w.t()

    with torch.no_grad():
        ref_gelu = F.gelu(F.linear(x.float(), w.float(), b.float()))
        ref_mm = x.float() @ wt.float()

    contestants = [
        ("Floor", "library GEMM only", lambda: x @ wt, ref_mm, 0.0),
        ("Your fused kernel", "Triton GEMM + bias + exact GELU epilogue", lambda: gemm_v5_gelu(x, wt, b), ref_gelu, 0.0),
    ]
    for name, what, lam, kw in COMPILED:
        fn = torch.compile(lam, **kw)
        print(f"compiling: {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                fn(x, w, b)                       # compile (+ autotune) happens on this first call
            sync()
        except Exception as e:
            print(f"  FAILED to compile: {type(e).__name__}: {str(e)[:300]}")
            contestants.append((name, what, None, ref_gelu, time.perf_counter() - t0))
            continue
        compile_s = time.perf_counter() - t0
        print(f"  done in {compile_s:.0f} s", flush=True)
        contestants.append((name, what, (lambda fn=fn: fn(x, w, b)), ref_gelu, compile_s))

    rows = []
    for name, what, fn, ref, compile_s in contestants:
        row = dict(name=name, what=what, compile_s=round(compile_s, 1), ok=False,
                   kernels=None, kernel_names=[], verdict="", ms=None, lo=None, hi=None)
        if fn is None:
            row["verdict"] = "failed to compile"
            rows.append(row); continue
        with torch.no_grad():
            try:
                torch.testing.assert_close(fn().float(), ref, atol=2e-2, rtol=2e-2)
                row["ok"] = True
            except AssertionError as e:
                row["verdict"] = "WRONG RESULT: " + str(e).splitlines()[0]
            row["kernels"], row["kernel_names"] = kernels_per_call(fn)
            runs = [time_ms(fn) for _ in range(5)]
        row.update(ms=statistics.median(runs), lo=min(runs), hi=max(runs))
        if row["ok"]:
            row["verdict"] = describe(row["kernels"], row["kernel_names"])
        rows.append(row)
        print(f"{name:36s} {row['ms']:.3f} ms  {'ok' if row['ok'] else 'WRONG'}  {row['verdict']}", flush=True)

    t = {r["name"]: r["ms"] for r in rows if r["ms"] is not None}
    flop = 2 * M * N * K
    print(f"\n### Free-fix test: {M} x {N} x {K}, bf16, exact GELU\n")
    print("| Contestant | What runs | Kernels | Time (ms) | Range (ms) | vs floor | TFLOPS |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r["ms"] is None:
            print(f"| {r['name']} | {r['what']} | - | failed | - | - | - |")
            continue
        k = r["kernels"] if r["kernels"] is not None else "?"
        print(f"| {r['name']} | {r['what']} | {k} | {r['ms']:.3f} | {r['lo']:.3f} to {r['hi']:.3f} "
              f"| {r['ms'] / t['Floor']:.2f}x | {flop / r['ms'] / 1e9:.1f} |")

    print("\nWhat each one ran:")
    for r in rows:
        names = ", ".join(n[:70] for n in r["kernel_names"]) or "-"
        print(f"  {r['name']}: {r['verdict']}\n      kernels: {names}")
    print("\nCompile + autotune time (first call):")
    for r in rows[2:]:
        print(f"  {r['name']}: {r['compile_s']} s")

    compiled_ok = {r["name"]: r["ms"] for r in rows[2:] if r["ok"]}
    fused = next(r for r in rows if r["name"] == "Your fused kernel")
    print("\nVerdict: " + (decide(fused["ms"], compiled_ok) if fused["ok"]
                           else "your fused kernel failed correctness; fix that first."))

    out = f"results/free_fix_M{M}.json"
    json.dump(dict(M=M, N=N, K=K, dtype="bf16", rows=rows), open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()