import csv, sys

def load(path):
    with open(path) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return {(r["M"], r["N"], r["K"]): r for r in rows}

base = load("results/rocblas.csv")
others = [load(p) for p in sys.argv[1:]]
for key, r in base.items():
    ref = float(r["tflops"])
    line = f"{r['tag']:8s} {key[0]:>6}x{key[1]:>6}x{key[2]:>6}  rocBLAS {ref:7.2f}"
    for d in others:
        t = float(d[key]["tflops"])
        line += f"   {d[key]['impl']} {t:7.2f} ({100 * t / ref:4.0f}%)"
    print(line)