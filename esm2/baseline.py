import argparse, json, os, time, torch
from transformers import AutoTokenizer, EsmModel
from esm2.flops import flops_for_tokens

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["3b", "650m"], default="3b")
args = p.parse_args()

PATHS = {"3b": "$PROJ/models/esm2_3b", "650m": "$PROJ/models/esm2_650m"}
PATH = os.path.expandvars(PATHS[args.model])
tok = AutoTokenizer.from_pretrained(PATH)
model = EsmModel.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

seqs = [s[:1022] for s in open("data/sprot_10k.txt").read().split()][:2000]   # 2,000 for speed
B = 16
batches = [seqs[i:i + B] for i in range(0, len(seqs), B)]

def run_batch(batch):
    enc = tok(batch, return_tensors="pt", padding=True)        # pads to the longest in the batch
    lens = enc["attention_mask"].sum(dim=1).tolist()            # real tokens, counted on the CPU
    padded_len = enc["attention_mask"].shape[1]
    model(**{k: v.cuda() for k, v in enc.items()})
    return lens, padded_len

with torch.inference_mode():
    for b in batches[:3]:                                       # warm-up, not timed
        run_batch(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    real = padded = useful = computed = 0
    t0 = time.perf_counter()
    for b in batches:
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
}
print(json.dumps(res, indent=2))
json.dump(res, open(f"results/esm_m1_baseline_{args.model}.json", "w"), indent=2)