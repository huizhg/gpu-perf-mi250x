import json
from esm2.flops import flops_for_protein, MODELS

peak = json.load(open("results/ceilings.json"))["matmul_TFLOPS"] * 1e12   # your measured roof
seqs = open("data/sprot_10k.txt").read().split()
for name, (layers, d) in MODELS.items():
    total = sum(flops_for_protein(len(s), name) for s in seqs)
    attn = sum(4 * layers * (min(len(s), 1022) + 2) ** 2 * d for s in seqs)
    seconds = total / peak
    print(f"{name:5s} total {total / 1e15:6.2f} PFLOP | attention share {attn / total:5.1%} | "
          f"speed of light {seconds:6.0f} s = {10_000 / seconds * 3600:,.0f} proteins per GPU-hour")