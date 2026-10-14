import argparse, json, os, time, torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, EsmModel
from esm2.flops import flops_for_tokens
from esm2 import rotary_patch

rotary_patch.apply()   # monkey-patch the rotary embedding to avoid Python-side cache

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["3b", "650m"], default="3b")
p.add_argument("--max-tokens", type=int, default=16384, help="max tokens per batch (batch size x longest length)")
p.add_argument("--compile", choices=["none", "static", "dynamic"], default="dynamic")
p.add_argument("--pipeline", choices=["sync", "pretok", "prefetch"], default="prefetch",
               help="sync: tokenize each batch inside the timed loop (no overlap); "
                    "pretok: tokenize everything before the timer (tokenization is free, the ceiling); "
                    "prefetch: tokenize the next batch in a background thread while the GPU works")
p.add_argument("--run", type=int, default=1, help="run index, so repeated runs don't overwrite each other")
args = p.parse_args()

PATHS = {"3b": "$PROJ/models/esm2_3b", "650m": "$PROJ/models/esm2_650m"}
PATH = os.path.expandvars(PATHS[args.model])
tok = AutoTokenizer.from_pretrained(PATH)
model = EsmModel.from_pretrained(PATH, dtype=torch.bfloat16, add_pooling_layer=False).cuda().eval()

if args.compile == "static":
    model = torch.compile(model, dynamic=False)
elif args.compile == "dynamic":
    model = torch.compile(model, dynamic=True)

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

def prep(batch, pin):
    """Tokenize one batch and count its real tokens on the CPU. Pinned memory allows an async copy."""
    enc = tok(batch, return_tensors="pt", padding=True)
    lens = enc["attention_mask"].sum(dim=1).tolist()
    padded_len = enc["attention_mask"].shape[1]
    if pin:
        enc = {k: v.pin_memory() for k, v in enc.items()}
    return enc, lens, padded_len

# pretok: tokenize every batch up front, outside all timing (as the day 3 baseline did)
pretokenize_s = None
if args.pipeline == "pretok":
    t_tok = time.perf_counter()
    pretokenized = [prep(b, pin=False) for b in batches]
    pretokenize_s = time.perf_counter() - t_tok

pool = ThreadPoolExecutor(max_workers=1)

def batch_stream():
    """Yields (inputs, lens, padded_len) per batch, produced the way the chosen pipeline produces them."""
    if args.pipeline == "sync":
        for b in batches:
            yield prep(b, pin=False)                       # the GPU waits while the CPU tokenizes
    elif args.pipeline == "pretok":
        yield from pretokenized                            # already done before the timer started
    else:
        future = pool.submit(prep, batches[0], True)
        for i in range(len(batches)):
            item = future.result()
            if i + 1 < len(batches):
                future = pool.submit(prep, batches[i + 1], True)   # tokenize the next batch while the GPU works
            yield item

def run_pass():
    """One pass over all batches, counting real, padded, useful and computed work."""
    real = padded = useful = computed = 0
    non_blocking = args.pipeline == "prefetch"             # only pinned memory makes this truly async
    for enc, lens, Lp in batch_stream():
        model(**{k: v.cuda(non_blocking=non_blocking) for k, v in enc.items()})
        real += sum(lens)
        padded += len(lens) * Lp
        useful += sum(flops_for_tokens(L, args.model) for L in lens)     # work the proteins needed
        computed += len(lens) * flops_for_tokens(Lp, args.model)         # work the GPU actually did
    return real, padded, useful, computed

with torch.inference_mode():
    t_c = time.perf_counter()
    run_pass()                                    # warm-up and compile, not timed
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t_c
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    real, padded, useful, computed = run_pass()   # timed
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

pool.shutdown()

peak = json.load(open("results/ceilings.json"))["matmul_TFLOPS"]
res = {
    "model": args.model,
    "pipeline": args.pipeline,
    "run": args.run,
    "num_proteins": len(seqs),
    "timed_pass_s": round(dt, 2),
    "proteins_per_gpu_hour": round(len(seqs) / dt * 3600),
    "padding_waste": round(1 - real / padded, 3),
    "useful_TFLOPS": round(useful / dt / 1e12, 1),
    "computed_TFLOPS": round(computed / dt / 1e12, 1),
    "useful_MFU": round(useful / dt / 1e12 / peak, 3),
    "peak_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    "attention_path": model.config._attn_implementation,
    "max_tokens": args.max_tokens,
    "num_batches": len(batches),
    "compile": args.compile,
    "compile_and_first_pass_s": round(compile_s, 1),
    "pretokenize_s": None if pretokenize_s is None else round(pretokenize_s, 2),
}
print(json.dumps(res, indent=2))
json.dump(res, open(f"results/esm_m2_{args.pipeline}_{args.compile}_run{args.run}.json", "w"), indent=2)