sbatch --account=project_462001433 --partition=small-g --nodes=1 \
       --gpus-per-node=1 --cpus-per-task=7 --mem=60G --time=02:00:00 \
       --wrap="sleep infinity"

## after the job is running, attach to it with:
# srun --jobid=<JOBID> --overlap --pty bash