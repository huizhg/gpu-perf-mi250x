import argparse, json, os, time, torch, random
from transformers import AutoTokenizer, EsmModel
from esm2.flops import flops_for_tokens
from esm2 import rotary_patch
# --- load the 3B model and tokenizer exactly as in bucketed.py ---
p = argparse.ArgumentParser()
p.add_argument("--model", choices=["3b", "650m"], default="3b")
p.add_argument("--max-tokens", type=int, default=8192, help="max tokens per batch (batch size x longest length)")
p.add_argument("--compile", choices=["none", "static", "dynamic"], default="dynamic")
args = p.parse_args()

PATHS = {"3b": "$PROJ/models/esm2_3b", "650m": "$PROJ/models/esm2_650m"}
PATH = os.path.expandvars(PATHS[args.model])
tok = AutoTokenizer.from_pretrained(PATH)
model = EsmModel.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

random.seed(0)
AA = "ACDEFGHIKLMNPQRSTVWY"
seqs = ["".join(random.choices(AA, k=n)) for n in (120, 300)]
inp = tok(seqs, return_tensors="pt", padding=True).to("cuda")

with torch.inference_mode():
    ref = model(**inp).last_hidden_state          # original, cached forward
    rotary_patch.apply()
    new = model(**inp).last_hidden_state          # patched, stateless forward

torch.testing.assert_close(new, ref, rtol=0, atol=0)
print("identical:", tuple(new.shape))