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

batches = [[seqs[i] for i in b] for b in token_budget_batches(seqs, args.max_tokens)]

encoded = [tok(b, return_tensors="pt", padding=True) for b in batches]   # all tokenization, untimed
# ...then, inside the timed loop:
#     model(**{k: v.cuda() for k, v in enc.items()})

def run_batch(encoded_batch):
    #enc = tok(batch, return_tensors="pt", padding=True)        # pads to the longest in the batch
    lens = encoded_batch["attention_mask"].sum(dim=1).tolist()            # real tokens, counted on the CPU
    padded_len = encoded_batch["attention_mask"].shape[1]
    model(**{k: v.cuda() for k, v in encoded_batch.items()})
    return lens, padded_len

with torch.inference_mode():
    for b in encoded[:3]:                                       # warm-up, not timed
        run_batch(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    real = padded = useful = computed = 0
    t0 = time.perf_counter()
    for b in encoded:
        lens, Lp = run_batch(b)
        real += sum(lens)
        padded += len(lens) * Lp
        useful += sum(flops_for_tokens(L, args.model) for L in lens)     # work the proteins needed
        computed += len(lens) * flops_for_tokens(Lp, args.model)         # work the GPU actually did
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

peak = json.load(open("results/ceilings.json"))["matmul_TFLOPS"]
res = {
    "model": args.model,
    "proteins_per_gpu_hour": round(len(seqs) / dt * 3600),
    "padding_waste": round(1 - real / padded, 3),
    "useful_TFLOPS": round(useful / dt / 1e12, 1),
    "computed_TFLOPS": round(computed / dt / 1e12, 1),
    "useful_MFU": round(useful / dt / 1e12 / peak, 3),
    "peak_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    "attention_path": model.config._attn_implementation,
    "max_tokens": args.max_tokens,
    "num_batches": len(batches),
}
print(json.dumps(res, indent=2))
json.dump(res, open(f"results/esm_m2_bucketed_test_tokenization_time{args.model}_{args.max_tokens}.json", "w"), indent=2)