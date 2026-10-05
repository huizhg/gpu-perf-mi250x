import csv
import matplotlib.pyplot as plt

def tflops(path, M, N, K):
    with open(path) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            if (int(r["M"]), int(r["N"]), int(r["K"])) == (M, N, K):
                return float(r["tflops"])

labels = ["v0\n32x32 tiles", "v1\n128x128", "v3\nautotuned", "v4\n+ MFMA 16", "rocBLAS"]
files = ["triton_v0", "triton_v1", "triton_v3", "triton_v4", "rocblas"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, n in enumerate((4096, 8192)):
    vals = [tflops(f"results/{f}.csv", n, n, n) for f in files]
    xs = [x + i * 0.4 for x in range(len(files))]
    bars = ax.bar(xs, vals, width=0.4, label=f"{n}³")
    ax.bar_label(bars, fmt="%.0f", fontsize=8)
ax.set_xticks([x + 0.2 for x in range(len(files))], labels)
ax.set_ylabel("TFLOPS (FP16, one MI250X GCD)")
ax.set_title("From first kernel to rocBLAS")
ax.legend()
fig.tight_layout()
fig.savefig("plots/progression.png", dpi=150)