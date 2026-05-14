#!/bin/bash
#SBATCH -p genoa
#SBATCH --array=0-99            # 100 array elements
#SBATCH --nodes=1               # Each array job uses 1 node
#SBATCH --ntasks-per-node=3     # 3 ecodag runs per node
#SBATCH --cpus-per-task=64      # 64 CPUs per ecodag run
#SBATCH -t 01:00:00
#SBATCH --output=slurm-%A.out   # Single output file for all array tasks

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

max_parallel=$SLURM_NTASKS_PER_NODE
running=0

for idx in $(seq "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" 2044); do
    out="output/results_avg_8_${idx}.csv"

    if [[ -s "$out" ]]; then
        echo "Skipping idx ${idx}: ${out} already exists"
        continue
    fi

    echo "Starting idx ${idx}"

    srun --exclusive -N1 -n1 -c "$SLURM_CPUS_PER_TASK" ./build/ecodag 8 "$idx" &

    running=$((running + 1))

    if (( running >= max_parallel )); then
        wait
        running=0
    fi
done

wait
