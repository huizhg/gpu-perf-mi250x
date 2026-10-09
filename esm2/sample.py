import gzip, random, statistics

seqs, buf = [], []
with gzip.open("data/uniprot_sprot.fasta.gz", "rt") as f:
    for line in f:
        if line.startswith(">"):           # a header line starts a new protein
            if buf:
                seqs.append("".join(buf))
            buf = []
        else:
            buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))

random.seed(0)                              # same sample every run
sample = random.sample(seqs, 10_000)
lens = sorted(len(s) for s in sample)
q = lambda p: lens[int(p * (len(lens) - 1))]
print(f"Swiss-Prot: {len(seqs):,} proteins; sample: {len(sample):,}")
print(f"length mean {statistics.mean(lens):.0f}, median {q(0.5)}, p90 {q(0.9)}, p99 {q(0.99)}, max {lens[-1]}")
print(f"longer than 1022 (truncated by ESM-2): {sum(l > 1022 for l in lens) / len(lens):.1%}")
open("data/sprot_10k.txt", "w").write("\n".join(sample))