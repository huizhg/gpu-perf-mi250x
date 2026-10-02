import csv, json, sys
import numpy as np
import matplotlib.pyplot as plt

ceil = json.load(open("results/ceilings.json"))
BW, PEAK = ceil["bandwidth_TBs"], ceil["matmul_TFLOPS"]
RIDGE = PEAK / BW

def load(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))

files = sys.argv[1:-1]            # result CSVs
out = sys.argv[-1]                # output PNG

fig, ax = plt.subplots(figsize=(8, 5.5))
ai = np.logspace(-0.5, 4, 300)
ax.plot(ai, np.minimum(PEAK, ai * BW), color="black", lw=2, label="measured roof")
ax.plot(ai, np.minimum(191.5, ai * 1.6), color="gray", ls="--", lw=1, label="datasheet roof")
ax.axvline(RIDGE, color="gray", ls=":", lw=1)
ax.text(RIDGE * 1.1, PEAK * 0.03, f"ridge ≈ {RIDGE:.0f} FLOP/B", color="gray")

markers = {"square": "o", "prefill": "s", "decode": "v", "esm": "D", "odd": "x"}
for path in files:
    rows = load(path)
    impl = rows[0]["impl"]
    for tag, m in markers.items():
        pts = [r for r in rows if r["tag"] == tag]
        if pts:
            ax.scatter([float(r["ai"]) for r in pts], [float(r["tflops"]) for r in pts],
                       marker=m, s=45, label=f"{impl} · {tag}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Performance (TFLOPS)")
ax.set_title(f"FP16 GEMM on one MI250X GCD  (BW {BW} TB/s, peak {PEAK} TFLOPS measured)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)