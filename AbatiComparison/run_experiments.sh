#!/bin/bash
#SBATCH -A m3058
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --output=output.txt
#SBATCH --error=error.txt

module load cgpu
export SLURM_CPU_BIND="cores"
module load python
source activate pytorch

srun python -u run.py