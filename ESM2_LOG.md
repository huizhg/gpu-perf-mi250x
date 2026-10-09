## M1
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
- Speed of light, 650M: 360,116 proteins/GPU-hour (attention share 6.2%)
- Baseline, 3B: 55907 proteins/GPU-hour (27.48% of speed of light), padding waste  60.52%, useful MFU: 0.275
- Baseline, 650M: 176656 proteins/GPU-hour (20.54% of speed of light), useful MFU 0.206
- Which model is more efficient, and why: 3B model. The useful work take more portion than the fixed cpu work, such as tokenization and kernel launches. 
- Two guesses for the missing time: 1. tokenization 2. kernel launches.
- Surprised me: nothing. 