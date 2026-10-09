#!/bin/bash
#SBATCH --job-name=esm-profile
#SBATCH --account=project_462001433
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=small-g
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# Capture ONE clean GPU kernel trace from a single-process vLLM latency run,
# then automatically convert it to a Perfetto (.pftrace) file.
#
# Why this works: `vllm bench latency` is a single offline process that loads
# the model, runs a few inferences, and exits on its own. No server, no forks
# to confuse the profiler, no kill/shutdown race. rocprofv3 captures it and
# writes a .db; we then convert that .db to .pftrace in the same job.
# ---------------------------------------------------------------------------

# No `set -e`: don't let the script bail out before the trace is written.

#MODEL=/scratch/project_462001433/hui/vLLM/models/Qwen2.5-VL-7B-Instruct
export PROJ=/scratch/project_462001433/hui/gpu-perf
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
export TRITON_CACHE_DIR=$PROJ/.triton_cache
export MIOPEN_USER_DB_PATH=$PROJ/.miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MPLBACKEND=Agg
export HF_HOME=/scratch/project_462001433/hf-cache

cd $PROJ
# Path that provides libdw.so.1 (needed by BOTH rocprofv3 and rocpd)
#ROCPROF_LIBS=/appl/local/csc/soft/eng/elmer/rocm-afar-8873-drop-22.2.0/lib/rocprofiler-systems

# Where the rocpd Python module lives inside the container
#ROCPD_PYPATH=/opt/rocm-7.0.2/lib/python3.12/site-packages

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

singularity exec --bind /scratch/project_462001433:/scratch/project_462001433 \
    --env HF_HOME=/scratch/project_462001433/hf-cache \
    "$SIF" "$@"\
    python -m esm2.baseline -- model 3b



