"""Compare the three tokenization pipelines.

Run from the repo root:  run python -m esm2.prefetch_summary                 (dynamic compile)
                         run python -m esm2.prefetch_summary --compile none   (eager)
"""
import argparse, glob, json, statistics

p = argparse.ArgumentParser()
p.add_argument("--compile", default="dynamic", choices=["none", "static", "dynamic"])
args = p.parse_args()

t, rng, rows_by = {}, {}, {}
for pl in ("pretok", "sync", "prefetch"):
    files = sorted(glob.glob(f"results/esm_m2_{pl}_{args.compile}_run*.json"))
    rows = [json.load(open(f)) for f in files]
    if not rows:
        print(f"{pl:9s} no runs found")
        continue
    rows_by[pl] = rows
    n = rows[0].get("num_proteins", 2000)
    secs = [n * 3600 / r["proteins_per_gpu_hour"] for r in rows]          # timed pass, seconds
    t[pl] = statistics.median(secs)
    rng[pl] = max(secs) - min(secs)
    runs = [r.get("run", "?") for r in rows]
    print(f"{pl:9s} runs {runs}  median {n * 3600 / t[pl]:,.0f} proteins/GPU-hour = {t[pl]:.2f} s"
          f"  (spread {rng[pl]:.2f} s)")

if {"pretok", "sync", "prefetch"} <= t.keys():
    noise = max(rng.values())
    stall = t["sync"] - t["pretok"]
    print(f"\nTokenization stall (sync - pretok):  {stall:+.2f} s   (largest run-to-run spread: {noise:.2f} s)")
    print(f"Prefetch vs sync:                    {t['sync'] / t['prefetch']:.3f}x")
    print(f"Prefetch vs pretok:                  {t['pretok'] / t['prefetch']:.3f}x")
    if abs(stall) <= noise:
        print("-> sync and pretok differ by less than the spread: no measurable stall to hide.")
    else:
        print(f"-> share of the stall prefetch hid: {100 * (t['sync'] - t['prefetch']) / stall:.0f}%")

# Can tokenization overlap with GPU work at all? Compare the two per batch.
pt = rows_by.get("pretok", [])
if pt and pt[0].get("pretokenize_s") is not None and "pretok" in t:
    nb = pt[0]["num_batches"]
    tok_ms = statistics.median(r["pretokenize_s"] for r in pt) / nb * 1e3
    gpu_ms = t["pretok"] / nb * 1e3
    print(f"\nPer batch ({nb} batches): tokenize {tok_ms:.1f} ms on the CPU vs {gpu_ms:.0f} ms on the GPU.")
    if tok_ms < gpu_ms:
        print("Tokenizing batch i+1 fits inside batch i's GPU time, so async kernel launches already hide it,")
        print("unless something in the forward pass makes the CPU wait for the GPU.")