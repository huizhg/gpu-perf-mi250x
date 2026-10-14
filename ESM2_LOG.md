## M1 day 1
### Sample 10,000 data from the Swiss-Prof dataset
Randomly sample 10,000 proteins from dataset Swiss-Prof, which has approximately half a million proteins. 
Statistics for the 10,000 sampled protein: 

Swiss-Prot: 575,748 proteins; sample: 10,000
length mean 368, median 294, p90 682, p99 1600, max 10498
longer than 1022 (truncated by ESM-2): 3.6%

### Model Anatomy and the Speed of light

#### number of parameters, layers, attention heads, FFN layers for two models, esm-2 650M and 3B
esm2_3b | layers 36 hidden 2560 heads 40 ffn 10240
   position embedding rotary | attention path sdpa
   parameters 2839 M
Some weights of EsmModel were not initialized from the model checkpoint at /scratch/project_462001433/hui/gpu-perf/models/esm2_650m and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
esm2_650m | layers 33 hidden 1280 heads 20 ffn 5120
   position embedding rotary | attention path sdpa
   parameters 651 M

#### The flops model and the speed of light
This is a ceiling that nobody can reach. "run python -m esm2.speed_of_light"

3b    total  20.40 PFLOP | attention share  3.2% | speed of light    177 s = 203,447 proteins per GPU-hour
650m  total   4.83 PFLOP | attention share  6.2% | speed of light     42 s = 860,116 proteins per GPU-hour

#### Memory for these two models: 
Weights in bf16: 3B is about 2.8B × 2 bytes ≈ 5.7 GB; 650M is about 1.3 GB. Both fit easily in one GCD's 64 GB. The biggest temporary buffer, if attention runs the plain way, is batch × heads × L² × 2 bytes per layer. Compute it for batch 16 at L = 1024 for both models (40 heads vs 20).

### The baseline pipeline
The baseline: take the proteins in file order, 16 at a time, pad each batch to its longest member, run the model. The simplest implementaion but bad. 
Results from the 3B model: 
This line is always printed out when i call the model: "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference." 

{
  "model": "3b",
  "proteins_per_gpu_hour": 55907,
  "padding_waste": 0.595,
  "useful_TFLOPS": 31.7,
  "computed_TFLOPS": 80.3,
  "useful_MFU": 0.275,
  "peak_memory_GB": 7.5,
  "attention_path": "sdpa"
}

The following is the result from the 650M model:
{
  "model": "650m",
  "proteins_per_gpu_hour": 176656,
  "padding_waste": 0.595,
  "useful_TFLOPS": 23.7,
  "computed_TFLOPS": 61.4,
  "useful_MFU": 0.206,
  "peak_memory_GB": 2.2,
  "attention_path": "sdpa"
}

### summary of M1: 
1. Proteins per gpu hour for 3B model: 55907 vs speed of light: 203,447 , 27.48% of the ceiling
  proteins per gpu hour for 650M model: 176656 vs speed of light: 860,116,  20.54% of the ceiling

2. Padding waste. what share of the gpu's work went into padding?  useful_TFLOPs vs computed_TFLOPS. wasted ration = (computed - useful) / computed. For 3B model, the number is : 60.52% . for 650M is : 61.4%
3. Even the computed_TFLOPS is below the GEMM roof. The time might go to small elementwise kernel and tokenization on the cpu between batches. 
4. The bigger model, 3B model has higher useful MFU. as the 3B model has bigger GEMM (2560 and 10240 vs 1280 and 5120). The tokenization and kernel launches are fixed work for both models, but 3B model do more work (bigger matmul), 

### End of ESM day 1
- Application submitted: yes
- Swiss-Prot sample: mean  368, median 294, p99 1600, >1022: 3.6%
- Attention path: sdpa
- Speed of light, 3B: 203,447 proteins/GPU-hour (attention share 3.2%)
- Speed of light, 650M: 860,116 proteins/GPU-hour (attention share 6.2%)
- Baseline, 3B: 55907 proteins/GPU-hour (27.48% of speed of light), padding waste  60.52%, useful MFU: 0.275
- Baseline, 650M: 176656 proteins/GPU-hour (20.54% of speed of light), useful MFU 0.206
- Which model is more efficient, and why: 3B model. The useful work take more portion than the fixed cpu work, such as tokenization and kernel launches. 
- Two guesses for the missing time: 1. tokenization 2. kernel launches.
- Surprised me: nothing. 

## M2 day 2

### the prediction before measurement
Try using bucketing to remove all padding. how much the performance would improve? now the padding takes around 60% of the useless computing. the useful throughput should rise 80.3 / 31.7 = 2.5 times. Roughly 140,000 proteins per GPU hours for the 3B model. but real bucketing leaves a little padding, and smaller batches of short proteins may run less efficiently. it is might be a little bit less than 2.5 times.

###  token-budget buckeing
The idea is to sort proteins by length, then fill each batch until it holds a fixed number of tokens (approximately). Short proteins will be packed in a bigger batch together, longer ones in small batches, and each batch is padded only to its own longest member.  A fixed token budget, rather than a fixed batch size, keeeps each batch's work and memory roughly constant. 
The follwings are the results from three budgets on the 3B model: we tested max_tokens for 8192, 16384, 32768. (if we count each protein has 1022 amino acids, whichi is the context window of esm-2, these three numbers are approximately 8192/1022=8, 16384/1022=16, 32768/1022=32 proteins per batch). but as most proteins are only hundreds of amino acids, one batch has more proteins than the estimation. 
{
  "model": "3b",
  "proteins_per_gpu_hour": 137195,
  "padding_waste": 0.016,
  "useful_TFLOPS": 77.9,
  "computed_TFLOPS": 79.2,
  "useful_MFU": 0.675,
  "peak_memory_GB": 6.6,
  "attention_path": "sdpa",
  "max_tokens": 8192,
  "num_batches": 89
}
{
  "model": "3b",
  "proteins_per_gpu_hour": 135899,
  "padding_waste": 0.031,
  "useful_TFLOPS": 77.1,
  "computed_TFLOPS": 79.6,
  "useful_MFU": 0.669,
  "peak_memory_GB": 7.5,
  "attention_path": "sdpa",
  "max_tokens": 16384,
  "num_batches": 45
}

{
  "model": "3b",
  "proteins_per_gpu_hour": 133485,
  "padding_waste": 0.056,
  "useful_TFLOPS": 75.8,
  "computed_TFLOPS": 80.4,
  "useful_MFU": 0.657,
  "peak_memory_GB": 9.1,
  "attention_path": "sdpa",
  "max_tokens": 32768,
  "num_batches": 23
}
From the above results, we can see that 8192 is the best budget that we have tested, now our useful_TFLOPS is 77.9, very close to 80.3. The computed TFLOPS changes when the batches are sorted, as the batches of short proteins are many small GEMMs. 

### Profile the bucketed pipeline
The the useless padding is gone, the next bottleneck becomes visible. Let's profile the bucketed pipeline and see where does the time go. confirm it with our two guesses: tokenizatio and kernel launches. we will test them both and check wich GPU operations take the time. 

For the bucketed pipeline, proteins per seccond: 
max_tokens      num_batches     proteins/s   bucket run total time   tokenize time
8192            89              38.10           52.49s                2.04s
16384           45              37.75           52.98s                2.04s
16384           23              37.08           53.93s                2.04s

So, we can see that the tokenizatio ntime is only uses around 3.7% of the total run time. 

Lets test where does the gpu time go. 

This is the result (too big, i didn't copy here), inside /results/esm_m2_profile.txt , instead, we can group the kernels into the following four buckets, in the following table, i'll show how much share of the gpu time of each bucket. 

This table mixes two kinds of rows:
Op rows (aten::...) are PyTorch operations. They have CPU time, because the CPU dispatches them.
Kernel rows (Cijk_..., attn_fwd, void at::native::...) are the GPU kernels those ops launch. Their Self CPU is 0.000us, which is how you spot them. 

When the model runs nn.Linear, the call chain looks like this:
aten::linear            (Self CUDA 0,        CUDA total 8.759 s)
  └─ aten::addmm        (Self CUDA 8.759 s,  CUDA total 8.759 s)
       └─ Cijk_..._Bias kernel  (8.758 s on the GPU)

Bucket      	          Ops counted	                                            Time	                        Share
GEMM	                       addmm	                                           8,759 ms	                      74.0%
Attention	      _efficient_attention_forward (attn_fwd)	                        422 ms                    	  3.6%
Elementwise and norm	  mul 829 + add 611 + erf 279 + native_layer_norm 200 	  1,920 ms	                    16.2%
Other	              cat 313 + rows below the top 20: 422	                      735 ms	                      6.2%
Total		                                                                        11,835 ms	                    100%
the attention path sdpa chose efficient_attention_forward  (attn_fwd)

How to read a trace in perfetto:
![How to read a trace in perfetto?](<./figures_for_esm_log/boundary_zoomed_annotated.png>)


![GPU kernel boundary between two consecutive batches (ESM-2 3B, bucketed pipeline, MI250X)](<./figures_for_esm_log/kernel boundaries between batches.png>)

The above trace shows the handover from batch N to batch N+1, about 0.25 ms of time. The top row is the CPU (Python main thread). The lower row is the GPU (stream 0).

Batch N ends on the GPU with a GEMM (Cijk_...) and an elementwise kernel (void...). The first flag marks the end of that last kernel. The second flag marks the first compute kernel of batch N+1. The gap between them is 0.13 ms. The GPU runs no compute during it. It only copies batch N+1’s two inputs from host to device (Memcpy HtoD, 12 µs and about 8 µs, likely input_ids and the attention mask), with about 30 µs of idle time between them.

On the CPU, the first aten::to calls hipMemcpyWithStream, which blocks for about 500 ms. Almost none of that is copying. The copy joins the same queue as the kernels, so it waits until batch N finishes on the GPU. The CPU call returns the moment the copy completes. After the second copy, the CPU starts the forward pass (aten::ne), and the first kernel follows.

After the first kernel, small kernels trickle in for another 0.2 ms. The GPU becomes continuously busy about 0.34 ms after the boundary.

### the Amdahl ranking

Why do we care about this? Because this is what guides us to decide what to work on. A three speed up on something takes only 2% of the time does't help with anything. a even 1.5x speed up on somthing takes 60% of the time saves a log.  
If a part takes a share of f of the total time, and you speed it up by s, then the time you save it: 
$f \times \left(1 - \frac{1}{s}\right)$

The problem is, how do we estimate "S"? how can we know how much we are able to speed some operations up by doing optimization? The followings show three ways to estimate s for different kind of work. 

1. Roofline headroom (for the compute bound work, like GEMM). s_max = roof/what we have achieved now. if a kernel already runs at 90%, then the best possible is 1/0.9 = 1.11

2. Memory-traffic counting. (for memory-bound work, like elementwise operations). These kernels do almost no math. Their time is the time to read and wirte the tensor in HBM. A chain of K unfused operations reads and writes the tensor k times; fused, it does so once. So s = bytes moved before/bytes moved after = k.

3. Hiding. (for work that can overlap). If a background worder can do the work while the GPU computes, it leaves the critical path entirely: s = $\infty$, and save time = f. 

Target	                    f	                      Fix	                        Achievable s	                Time saved
Elementwise and norm	     16.2%	      fusion (Inductor or Fusion lab)	          ~2.5	                        ~9.7%
CPU tokenization gaps    	~0.06%	        tokenize ahead in a background worker     	∞	                          ~3.7%
GEMMs	                     74.0%	       better kernels or tuning	                ~1.05 (max 1.09)	          ~3.5% (max 6%)
Attention	                  3.6%	        a faster fused kernel	                    ~1.2	                       ~0.6%

The above table shows our estimation of how much speed up we can have for each target. The followings are the reasonings of how we got these estimated number. 

Elementwise and norm, s ≈ 2.5 (method 2). The bucket is mostly unfused chains:

Tokenization, s = ∞ (method 3). The above test shows that the tokenization takes around 3.7% of the total time (2.04s/53s), so handling all the tokenization before will not stalling gpu, which will give us a 3.7% time saved. 


GELU runs as erf, mul and add kernels on the 4× hidden-width FFN tensor (10,240 wide), roughly 4 passes over the same data. Fused, it’s 1 pass: about 4×.
Rotary runs mul, cat, neg, add on q and k, about 5 passes. Fused, it’s 1: about 5×.
Residual add and LayerNorm are already single kernels, so they gain little unless fused with each other: about 1 to 2×.

Weighting these, about 2.5× for the whole bucket is a fair middle estimate, giving a 8 to 11% range for s between 2 and 3. Bonus: the cat in “Other” (2.6%) is part of the rotary chain, so fusion removes that too. That pushes f to about 19%.

GEMMs, s ≈ 1.05 (method 1). The best run computed 79.2 TFLOPS averaged over the whole run. Nearly all FLOPs are GEMMs, since attention FLOPs are about 2% at these lengths, but GEMMs take only 74% of the time. So the GEMM kernels themselves run at about 0.98 × 79.2 ÷ 0.74 ≈ 105 TFLOPS, about 91% of your 115 TFLOPS roof. The ceiling is 115 ÷ 105 ≈ 1.09, and realistic tuning gets part of it.

That’s why GEMM saves less than fusion despite being 4.6× bigger: it’s already near its roof.

Attention, s ≈ 1.2 (judgment). SDPA already uses a fused kernel (attn_fwd). A better one might gain 10 to 30%, and even infinite speedup would save only 3.6%.



### End of ESM day 2
- Prediction for bucketing: 2.5x
- Best budget: 8192 tokens -> 137195 proteins/GPU-hour (2.45x vs baseline), padding waste 1.6%, useful MFU 0.676
- Tokenization: 2.04s of 52.5 s total (3.8%)
- GPU time: GEMM 74%, attention 3.6%, elementwise/norm 16.2%, other 6.2%
- SDPA kernel: attn_fwd 
- Amdahl top target: elementwise and norm
- Surprised me: The difficulty for the Amdahl ranking estimation. 


## Day 3

### how much speed up are we expecting if we are using torch.compile?
If the Inductor speeds up the 16.2% elementiwise adn norm operations by 2.5x and change nothing else, the time drops by 0.162 x (1-1/2.5) = 9.7%. which is about 1.11 more proteins per gpu hour. if we are using the best bucket size we tested last time: 8192 tokens, it produces 137195 proteins per GPU hour. after the optimization, it should be around 150914 proteins per gpu hour.  We also tested if the tokenizatin stall the GPU. Yesterday, from the experiment, it showed that the tokenizatin takes 3.8% of the wall time when run on its own. Today, we tested it by computing all the tokennization in advance, before inference. the gpu wall time is reduced from 52.5s to 50.5s. exactly two seconds less. which means that the tokenization was stalling the GPU. From now on, all the experiment use the pre-tokenization setting, to avoid gpu stalling by CPU's tokenizatin work. 


### Compile the bucked pipeline
The pitfall of torch.compile. Torch.compile specializes the generated code to the input shapes it sees.  Our bucketed pipeline has 89 batches, and almost every one has a different shape (batch size x length). By default, each new shape triggers a new compilation, and after a small number of them (8 by default ) pytorch gives up and silenty falls back to eager mode for new shapes.  This is why torch.compile might look not helpful at all. There are two fixes for this: mark the shapes as dynamic, so one compiled graph handles any batch size and length, or round lengths up to a few standard sizes, so only a handful of shapes ever appear. 

Mode        Proteins per GPU-hour      vs eager       Compile + first pass(s)     ecompilations
Eager (None)       142268               1.00x           51.8                    0
Static             142269               1.00x           343.6                   8
Dynamic            167290               1.17x           140.2                   0

while doing this experiment, i've gotten some problems. both the static and dynamic compile mode fell back to eager mode after 8 compilation. 

- Static and dynamic both logged 8 recompilations, then `hit config.recompile_limit (8)`. After that, PyTorch ran every new shape in eager mode, so most of the 89 batches ran uncompiled and neither result measured Inductor.
- Static falling back was expected: almost every batch has a new shape. Dynamic falling back was not. Its recompile log showed the cause: every guard failure was on `rotary_embeddings._seq_len_cached` (modeling_esm.py:104).
- Cause: the Hugging Face ESM rotary embedding caches its cos/sin tables and stores the cached length in a Python integer on the module. torch.compile treats integer module attributes as constants, so each new sequence length broke the guard and forced a recompile, whatever `dynamic=True` said about the input tensors.
- Fix: replaced `RotaryEmbedding.forward` with a stateless version that rebuilds the tables on every call (`esm2/rotary_patch.py`). The cache never helped here anyway: lengths change at almost every batch. The patched model's output is bit-identical to the original (rtol=0, atol=0).
- Result after the patch: dynamic ___ recompilations, static ___ (expected to still hit the limit, now on input shapes).

### A fixed-shape microbenchark
The pipeline now mixes two questions: how good inductor's kernels are, and how well compilation copes with changing shapes. Now we are going to test it with a fixed shape. We will test with one batch, with shape [16, 512], which is exactly 8192 token budget.  this batch contains 16 random sqeuences of 510 residues (plus the first and end token is 512). Then we test the forward pass for each mode: eager, default and max-autotune. 

{'input_ids': (16, 512), 'attention_mask': (16, 512)}
eager           584.53 ms per forward
default         489.79 ms per forward
max-autotune    487.33 ms per forward

from the above result from the  Micro benchmark: (16 x 512): eager 584.5 ms, default 489.8 ms -> 1.19x. Pipeline dynamic: 1.18x.
Dynamic shapes kept ~91% of the kernel gain (17.6% of 19.3%); the ~1.5% gap is the most fixed-size buckets could recover. Not worth it. and autotune brought almost nothing to this shape. 0.5% over default. 


### read the inductor and find out what kernels are fused in the default mode (torch.compile, static):

cli output: 
      1 def triton_red_fused_add_addmm_native_layer_norm_view_9
      1 def triton_red_fused_add_addmm_native_layer_norm_view_8
      1 def triton_red_fused_add_addmm_native_layer_norm_view_7
      1 def triton_red_fused__to_copy_div_embedding_eq_masked_fill_mul_native_layer_norm_rsub_unsqueeze_1
      1 def triton_red_fused__to_copy_add_addmm_div_embedding_eq_masked_fill_mul_native_layer_norm_rsub_unsqueeze_view_5
      1 def triton_poi_fused_addmm_tanh_10
      1 def triton_poi_fused_add_addmm_div_erf_mul_view_6
      1 def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_arange_cat_cos_expand_mul_neg_rsub_sin_split_transpose_unsqueeze_view_4
      1 def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_arange_cat_cos_expand_mul_neg_rsub_sin_split_transpose_unsqueeze_view_3
      1 def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_addmm_arange_cat_cos_expand_mul_neg_rsub_sin_split_transpose_unsqueeze_view_2
      1 def triton_per_fused_eq_sum_0


| Kernel | What it fuses |
|---|---|
| `triton_per_fused_eq_sum_0` | Counts mask tokens per sequence (`input_ids == mask_id`, then sum), for ESM's token-dropout scaling |
| `triton_red_fused__to_copy_div_embedding_eq_masked_fill_mul_native_layer_norm_rsub_unsqueeze_1` | Embedding lookup, token-dropout rescaling (`masked_fill`, `mul`, `div`), attention-mask prep (`rsub` = 1 − mask) and the first LayerNorm. The whole input stage in one kernel |
| `triton_red_fused__to_copy_add_addmm_..._embedding_..._native_layer_norm_..._view_5` | Most likely the first layer's bias + residual add + LayerNorm, with the cheap embedding recomputed inside instead of stored and reread |
| `triton_poi_fused__scaled_dot_product_efficient_attention_..._arange_cat_cos_sin_..._transpose` (`_2`, `_3`, `_4`) | Rotary embedding: `arange`, `cos`, `sin` (the patched table rebuild), `cat`, `neg`, `mul`, `add`, plus the reshape and transpose into the head layout SDPA expects. Three variants; `_2` also absorbs a linear layer's bias |
| `triton_poi_fused_add_addmm_div_erf_mul_view_6` | GELU: the up-projection's bias add, then exact GELU, `x · 0.5 · (1 + erf(x / √2))`. Still a separate pass after the library GEMM |
| `triton_red_fused_add_addmm_native_layer_norm_view` (`_7`, `_8`, `_9`) | Bias of the preceding linear layer + residual add + next LayerNorm: after attention, after the FFN, and the final LayerNorm |
| `triton_poi_fused_addmm_tanh_10` | Pooler: bias + `tanh` on token 0 of each sequence. Its output is unused |


1. **Which fused kernels appear, and what does each combine?**
   - GELU: `triton_poi_fused_add_addmm_div_erf_mul_view_6` = up-projection bias + exact GELU (`erf`).
   - Residual + LayerNorm: `triton_red_fused_add_addmm_native_layer_norm_view` (x3) = bias + residual add + next LayerNorm (after attention, after FFN, final).
   - Also fused: the whole input stage (embedding, token-dropout scaling, mask prep, first LayerNorm) and the rotary embedding with the head transpose.

2. **Are the GEMMs still library calls?**
   - Yes. Default mode generates no Triton GEMMs (no `triton_tem_` kernels).
   - Inductor splits each `addmm` into a library matmul plus a bias add, and fuses the bias into the next kernel.
   - GELU stays a separate pass: it rereads the FFN output from memory. Fusing GELU into the GEMM epilogue is what the Fusion lab can do and Inductor's default mode doesn't.

3. **Does the rotary `cat` disappear into a fused kernel?**
   - Yes. It's inside the rotary kernels, together with `cos`, `sin`, `neg`, `mul`, `add` and the patched table rebuild.

### Profile the compile pipeline (torch.compile, dynamic)

| Bucket | Eager share | Compiled share | Eager time (ms) | Compiled time (ms) |
|---|---|---|---|---|
| GEMM | 74.0% | 86.3% | 8,759 | 8,722 |
| Attention | 3.6% | 4.9% | ~426 | 491 |
| Elementwise and norm | 16.2% | 8.8% | ~1,918 | 887 |
| Other | 6.2% | ~0.1% | ~734 | ~9 |
| **Total GPU time (20 batches)** | | | **~11,836** | **10,109** |

What the comparison shows:

GEMMs didn’t change (8,759 → 8,722 ms). Their share rose from 74% to 86% only because everything around them shrank.
Elementwise time fell 2.2× (1,918 → 887 ms), close to the s ≈ 2.5 you assumed.
“Other” almost vanished (734 → 9 ms). The dtype casts, copies and embedding lookup are now inside fused kernels: their names contain _to_copy and embedding. Your 1.11× prediction left this bucket out, and it’s where the extra 6% came from.
Total GPU time fell 1.17× (11,836 → 10,109 ms), matching the pipeline’s measured 1.18×. The profile and the stopwatch agree.
Attention rose 15% (426 → 491 ms). That’s 65 ms, small next to the rest. It may be node drift, like the 5% we saw this morning, or Inductor handing SDPA its inputs in a different memory layout. Note it as unexplained.
The kernel names match Step 4, so the line about fusion not depending on dynamic shapes holds.

The new Amdahl table (all shares of wall time, compiled pipeline, 43.0 s per 2,000 proteins):

| Target | f (wall time) | Fix | Achievable s | Time saved | Cost of the fix |
|---|---|---|---|---|---|
| CPU tokenization stalls | ~4.6% (2.0 s of 43.0 s) | tokenize ahead in a background thread, pinned memory | ∞ (fully hidden) | ~4.6% | low: ~20 lines |
| GEMMs | 82.3% (86.3% of GPU time) | better kernels or library tuning | ~1.05 (max 1.09) | ~3.9% (max 6.8%) | very high: already vendor-tuned |
| GELU pass | 3.5% (3.7% of GPU time) | fuse GELU into the GEMM epilogue (Fusion lab) | large, minus the erf cost in the epilogue | ≤3.5% (realistic ~2-3%) | medium: one kernel, per shape |
| Attention | 4.7% (4.9% of GPU time) | recover the compile regression (422 ms eager vs 491 ms compiled) | ~1.16 | ~0.7% | medium: investigate Q/K/V layout |
| Residual + LayerNorm | 2.8% (2.9% of GPU time) | already fused; LayerNorm needs a full-row reduction, hard to merge into a GEMM | ~1 | ~0% | n/a |
| Rotary + attention input prep | 2.0% (2.1% of GPU time) | already fused by Inductor | ~1 | ~0% | n/a |

### End of ESM day 3
- Prediction for compile: 1.11x
- Tokenization test: total 52.5 s -> 50.5 s with pre-tokenized input (stalling: yes, ~3.8%)
- Pipeline: eager 142,268, static 142,269, dynamic 167,290 proteins/GPU-hour (dynamic 1.18x vs eager); recompilations static 8 (hit the limit, fell back to eager), dynamic 0
- Micro (16 x 512): eager 584.53 ms, default 489.79 ms (1.19x), max-autotune 487.33 ms (1.20x)
- Inductor fused: bias + GELU into one pointwise kernel; residual add + bias + LayerNorm; rotary embedding + attention input prep; embedding + first LayerNorm; pooler dense + tanh (unused by the pipeline)
- GEMMs still library calls: yes (Tensile Cijk kernels, 86% of GPU time; no triton_tem_ kernels)
- GELU fused into GEMM: no (Inductor moved the bias out of addmm into a separate pointwise GELU kernel)
- Elementwise share: 16.2% eager -> 8.8% compiled (time ~1.92 s -> ~0.89 s, measured s ~2.2)
- Attention: 422 ms eager -> 491 ms compiled (slower; open question)
- New Amdahl top target: CPU tokenization stalls (~3.8%), then GELU-into-GEMM fusion (<=3.7%)
- Surprised me: static compile was exactly as fast as eager, because it silently fell back after 8 recompilations; and attention got slower after compiling


## Day 4

### tokenizer prefetch
we have found that the tokenization in cpu stalls the gpu. now we tokenize the next batch in a background thread while the gpu works. and put the result in pinned memory, which allows a asynchronous copy. we create a copy of the compiled.py and name it prefetch.py with dynamic compile. 
the result is : 166476 proteins per gpu hour. and the compile_and_first_pass_s is  133.7s for the first run. and then around 80s for the other two runs, as the first run is cold compile, the second and third run used the cache from the first one. Then I found a problem from the previous measurement. in the compiled.py. we put all the tokenization before the inference. Where the measuremnet for the dynamic compile actually give us the ceiling number: 167290. which is because the tokenization time is not counted in the proteins per gpu hours, as we treat tokenization for free, becuase we fed each batch the ready tokenized the embeddings. To make a fair comparision. we need to do another three run in a new prefetch.py, where we test three mode: pretok: all up front, out side the timer, which represent the celing. And sync: the tokenizatin is per batch, inside the timer, gpu waits. then prefetch: per batch, the tokenization run in the background thread. 

The new resulsts are: 
pretok: 165290 (ceiling)
Sync: 164048
Prefetch: 165180


### Fusion lab
Fusion lab prediction (before measuring)
Shape 8192 x 2560 -> 10240, bf16, exact GELU.
- GEMM: 2 x 8192 x 10240 x 2560 = 430 GFLOP, ~4.3 ms at 100 TFLOPS. Floor ~4.3 ms.
- Unfused GELU traffic: 8192 x 10240 x 2 B = 168 MB read + 168 MB written = 336 MB,
  0.27 ms at 1.25 TB/s. Profile shows 0.52 ms per call, so it runs at about half bandwidth.
- The fused kernel deletes that traffic but still computes erf.
- Prediction: fused = floor + a small epilogue cost, 0.3 to 0.5 ms faster than Inductor.



### Fusion lab: 8192 x 10240 x 2560, bf16, exact GELU

| Contestant | What runs | Kernels | Time (ms) | Range (ms) | vs floor | TFLOPS |
| --- | --- | --- | --- | --- | --- | --- |
| Floor | x @ w.t() only | 1 | 3.815 | 3.812 to 3.820 | 1.00x | 112.6 |
| Eager | addmm + separate GELU | 2 | 4.238 | 4.236 to 4.239 | 1.11x | 101.3 |
| Inductor default | mm + fused bias-and-GELU kernel | 2 | 4.259 | 4.257 to 4.263 | 1.12x | 100.8 |
| fused kernel | Triton GEMM + bias + exact GELU epilogue | 1 | 3.902 | 3.900 to 3.905 | 1.02x | 110.1 |
| GEMM alone | same kernel, epilogue off | 1 | 3.671 | 3.670 to 3.672 | 0.96x | 117.0 |

### Day 4, Step 3: Amdahl
1. Kernel: fused 3.902 ms, 2.3% above the floor (3.815). Beat Inductor by 0.358 ms (predicted 0.3-0.5).
   Split: 0.213 fusion (epilogue 0.231 replaces a 0.444 GELU pass) + 0.144 faster GEMM (3.671 vs 3.815).
   Prediction missed the erf compute: ~0.21 ms of vector work that fusion exposes instead of hiding.
2. Model: 3.7% x 0.358 / 0.444 = ~3.0% end to end (fusion alone ~1.8%). Upper bound: tuned at M = 8192 only.
3. Ship it? 
  For: 
    About 3% end to end, on the largest elementwise kernel left after compiling.
    Two kernels become one.
    The GEMM half could ship separately: a faster tile config than the library’s is worth contributing back even without the fusion.
    
  Against:
    Tuned for one shape on one GPU; MI300 has different matrix instructions and LDS sizes.
    Variable batch shapes put the 1.2% GEMM part at risk.
    The fusion alone is only 1.8%.
I   nductor may learn this epilogue fusion itself, which makes the maintenance cost temporary but real.

### End of ESM day 4
- - Prefetch: 166,476 proteins/GPU-hour (dynamic compile), ~1.00x vs sync. Null result: tokenizing a batch
  takes 46 ms vs ~960 ms (compiled) to 1,150 ms (eager) of GPU work, and async launches already overlap them.
  Eager check: sync 51.62 s vs pretok 51.58 s.
- Fusion lab (8192 x 10240 x 2560, bf16): floor 3.815 ms, eager 4.238 ms, Inductor 4.259 ms, fused 3.902 ms
- Your GEMM alone: 3.671 ms (104% of the floor's speed)
- Predicted saving vs Inductor: 0.3-0.5 ms; measured: 0.358 ms (0.213 fusion + 0.144 faster GEMM)
- End-to-end saving if deployed: ~3.0% (fusion alone ~1.8%; upper bound, tuned at M = 8192 only)
- Ship it? maybe not. Need to check if there's free fix from inductor. 
- Surprised me:
  1. The epilogue costs 0.231 ms, half a separate GELU pass. Fusion removed the memory traffic,
     but the erf compute (~0.2 ms of vector work) now runs while the matrix cores sit idle.
  2. My Triton GEMM beat hipBLASLt by 3.8% on this shape (117.0 vs 112.6 TFLOPS).
  3. Eager beat Inductor (4.238 vs 4.259 ms): hipBLASLt already fuses the bias,
     so Inductor's addmm split bought nothing here.
  4. Day 3's 2.0 s "tokenization stall" equals total tokenization time (2.05 s), so the day 3 script ran
     tokenization serially (suspect: a per-batch GPU-to-CPU sync, ___). Today's loop hides it with no prefetch.

## Day 5

- Free-fix test (one session; node ~3% slower than day 4, compare within): floor 3.950 ms, fused 4.010,
  Inductor default 4.403, max-autotune 4.229 (own Triton GEMM, GELU separate, 2 kernels; Triton-only identical).
  Max-autotune recovers the GEMM half (0.174 ms) for free; fusion half (0.219 ms) remains mine.
  Fused kernel 5.5% faster than the best compiler setting: earns its place at this shape.
  Open: why no epilogue fusion (hypothesis: epilogue benchmarked on 1 config only); dynamic-shape max-autotune untested.