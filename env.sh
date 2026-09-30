export PROJ=/scratch/project_462001433/hui/gpu-perf
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
export TRITON_CACHE_DIR=$PROJ/.triton_cache
export MIOPEN_USER_DB_PATH=$PROJ/.miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MPLBACKEND=Agg
run() { singularity exec --bind /scratch/project_462001433:/scratch/project_462001433 "$SIF" "$@"; }
cd $PROJ