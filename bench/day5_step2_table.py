"""Day 5 Step 2: build the kernel table from the rocprofv3 output.

Run from the gpu-perf folder:
    python bench/day5_table.py

Prints the Markdown table for LOG.md and also saves it to results/prof/day5_table.md.
"""
import csv
import re
from pathlib import Path

PROF = Path("results/prof")
HARNESS = {"rocblas": Path("results/rocblas.csv"),      # harness CSVs with an "ms" column
           "triton": Path("results/triton_v4.csv")}     # run_one.py uses gemm_v4
SHAPES = {"4096": (4096, 4096, 4096),
          "prefill": (2048, 11008, 4096),
          "decode": (1, 4096, 4096)}
TIMED = 50                                    # timed iterations in bench/run_one.py
HELPERS = ("__amd_rocclr_", "distribution_", "FillFunctor")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def is_helper(name):
    return any(h in name for h in HELPERS)


def main_kernel(stats):
    """The non-helper kernel with the largest total GPU time."""
    real = [r for r in stats if not is_helper(r["Name"])] or stats
    return max(real, key=lambda r: float(r["TotalDurationNs"]))


def timed_loop_avg_us(trace_path, name):
    """Average of the last TIMED dispatches of `name`, i.e. the timed loop only.

    Skips the warm-up, which for Triton also contains the autotuner trying configs.
    """
    if not trace_path.exists():
        return None
    rows = read_csv(trace_path)
    if not rows:
        return None
    cols = list(rows[0])
    name_col = next(c for c in cols if c.lower() in ("kernel_name", "name"))
    start_col = next(c for c in cols if c.lower().startswith("start"))
    end_col = next(c for c in cols if c.lower().startswith("end"))
    hits = sorted((int(r[start_col]), int(r[end_col]) - int(r[start_col]))
                  for r in rows if r[name_col] == name)
    last = [d for _, d in hits[-TIMED:]]
    return sum(last) / len(last) / 1000 if last else None


def harness_us(impl, shape):
    path = HARNESS[impl]
    if not path.exists():
        return None
    for r in read_csv(path):
        if (int(r["M"]), int(r["N"]), int(r["K"])) == shape:
            return float(r["ms"]) * 1000
    return None


def short(name, width=60):
    name = name.split("(")[0]
    if len(name) <= width:
        return name
    tile = re.search(r"_MT\d+x\d+x\d+(_MI\d+x\d+x\d+)?", name)
    return name[:30] + "..." + (tile.group(0) if tile else name[-20:])


def fmt(x):
    return f"{x:.2f}" if x is not None else "n/a"


table = ["| Run | Main kernel name | Calls | Average duration (µs) | Harness time (µs) | Harness vs kernel |",
         "|---|---|---|---|---|---|"]
notes = []

for shape_key, shape in SHAPES.items():
    for impl in ("rocblas", "triton"):
        run = f"{impl}_{shape_key}"
        stats_path = PROF / f"{run}_kernel_stats.csv"
        if not stats_path.exists():
            table.append(f"| {run} | missing {stats_path.name} | | | | |")
            continue

        stats = read_csv(stats_path)
        k = main_kernel(stats)
        stats_avg = float(k["AverageNs"]) / 1000
        loop_avg = timed_loop_avg_us(PROF / f"{run}_kernel_trace.csv", k["Name"])
        avg = loop_avg if loop_avg is not None else stats_avg
        h = harness_us(impl, shape)
        diff = f"{100 * (h - avg) / avg:+.1f}%" if h is not None else "n/a"
        table.append(f"| {run} | `{short(k['Name'])}` | {k['Calls']} | {fmt(avg)} | {fmt(h)} | {diff} |")

        notes.append(f"- **{run}** full kernel name: `{k['Name']}`")
        if loop_avg is not None and abs(loop_avg - stats_avg) / loop_avg > 0.05:
            notes.append(f"  - kernel_stats average is {stats_avg:.2f} µs over all {k['Calls']} calls; "
                         f"the table uses the last {TIMED} calls ({loop_avg:.2f} µs) because warm-up differs")
        others = [r for r in stats if r is not k and not is_helper(r["Name"]) and int(r["Calls"]) >= TIMED]
        for r in others:
            notes.append(f"  - also ran {r['Calls']}x, {float(r['AverageNs']) / 1000:.2f} µs avg: `{short(r['Name'])}`")
        if h is None:
            notes.append(f"  - no harness row for M,N,K = {shape} in {HARNESS[impl]}")

out = "\n".join(table) + "\n\nNotes:\n" + "\n".join(notes) + "\n"
print(out)
(PROF / "day5_table.md").write_text(out)
print(f"Saved to {PROF / 'day5_table.md'}")