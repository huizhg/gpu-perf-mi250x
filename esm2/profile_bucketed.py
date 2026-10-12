import argparse, json, os, time, torch
from transformers import AutoTokenizer, EsmModel
from esm2.flops import flops_for_tokens

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["3b", "650m"], default="3b")
p.add_argument("--max-tokens", type=int, default=16384, help="max tokens per batch (batch size x longest length)")
args = p.parse_args()

PATHS = {"3b": "$PROJ/models/esm2_3b", "650m": "$PROJ/models/esm2_650m"}
PATH = os.path.expandvars(PATHS[args.model])
tok = AutoTokenizer.from_pretrained(PATH)
model = EsmModel.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

seqs = [s[:1022] for s in open("data/sprot_10k.txt").read().split()][:2000]   # 2,000 for speed

def token_budget_batches(seqs, max_tokens):
    """Sort by length, then pack each batch up to max_tokens (batch size x longest length)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs[i]) + 2                      # residues + start and end tokens
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches                                # lists of indices into seqs

# now we test our best budget max_tokens = 8192 from the earlier resulst

max_tokens = 8192

batches = [[seqs[i] for i in b] for b in token_budget_batches(seqs, max_tokens)]

def run_batch(batch):
    enc = tok(batch, return_tensors="pt", padding=True)        # pads to the longest in the batch
    lens = enc["attention_mask"].sum(dim=1).tolist()            # real tokens, counted on the CPU
    padded_len = enc["attention_mask"].shape[1]
    model(**{k: v.cuda() for k, v in enc.items()})
    return lens, padded_len

from torch.profiler import profile, ProfilerActivity


with torch.inference_mode():
    for b in batches[:3]:                          # warm-up
        run_batch(b)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in batches[:20]:
            run_batch(b)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("results/esm_m2_trace.json")