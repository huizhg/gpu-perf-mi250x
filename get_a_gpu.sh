srun --account=project_462001433 --partition=small-g --nodes=1 \
     --gpus-per-node=1 --cpus-per-task=7 --mem=60G --time=02:00:00 --pty bash
source /scratch/project_462001433/hui/gpu-perf/env.sh