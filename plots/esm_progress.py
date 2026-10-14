import matplotlib.pyplot as plt

steps = ["Baseline\nbatch 16", "Bucketed\n8,192 tokens", "+ torch.compile\n(dynamic)"]
values = [55_907, 137_195, 167_290]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(steps, values, color=["#999999", "#4C72B0", "#DD8452"])
ax.bar_label(bars, labels=[f"{v:,}" for v in values], fontsize=9)
ax.axhline(203_447, ls="--", color="black", lw=1)
ax.text(2.45, 206_000, "speed of light", ha="right", fontsize=9)
ax.set_ylabel("Proteins per GPU-hour (ESM-2 3B, one MI250X GCD)")
ax.set_title("Embedding Swiss-Prot with ESM-2 3B: 2.99x from the pipeline and the compiler",
             fontsize=11)
fig.tight_layout()
fig.savefig("plots/esm_progress.png", dpi=150)