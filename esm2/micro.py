import argparse, json, os, time, torch,random
from transformers import AutoTokenizer, EsmModel
from esm2.flops import flops_for_tokens
from esm2 import rotary_patch
rotary_patch.apply()

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["3b", "650m"], default="3b")
p.add_argument("--max-tokens", type=int, default=8192, help="max tokens per batch (batch size x longest length)")
p.add_argument("--compile", choices=["none", "static", "dynamic"], default="none")
args = p.parse_args()

PATHS = {"3b": "$PROJ/models/esm2_3b", "650m": "$PROJ/models/esm2_650m"}
PATH = os.path.expandvars(PATHS[args.model])
tok = AutoTokenizer.from_pretrained(PATH)
model = EsmModel.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

if args.compile == "static":
    model = torch.compile(model, dynamic=False)
elif args.compile == "dynamic":
    model = torch.compile(model, dynamic=True)


import torch
from bench.harness import time_ms

AA = "ACDEFGHIKLMNPQRSTVWY"
def make_batch(B, L):                     # L counts the BOS and EOS tokens
    seqs = ["".join(random.choices(AA, k=L - 2)) for _ in range(B)]
    return tok(seqs, return_tensors="pt").to("cuda")

random.seed(0)
inp = make_batch(16, 512)                 # 16 x 510 residues + 2 special tokens = 8,192 tokens
print({k: tuple(v.shape) for k, v in inp.items()})   # expect (16, 512) for both

modes = {"eager": model,
         "default": torch.compile(model, dynamic=False),
         "max-autotune": torch.compile(model, mode="max-autotune", dynamic=False)}
with torch.inference_mode():
    for name, m in list(modes.items())[:2]: # skip max-autotune in this run. for saving the inductor code
        m(**inp)                          # compiles on the first call, untimed
        print(f"{name:13s} {time_ms(lambda: m(**inp)):8.2f} ms per forward")