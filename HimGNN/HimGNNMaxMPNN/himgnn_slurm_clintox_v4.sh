#!/bin/bash
#SBATCH -J himgnn_clintox_maxmpnn                 # Job name
#SBATCH --mail-user=aanaidu@stats.ox.ac.uk   # Your Stats email
#SBATCH --mail-type=BEGIN,END,FAIL     # Notifications
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=102400M                    # Memory request

echo "Running on $(hostname)"

source /vols/teaching/msc-projects/2024-2025/aanaidu/yes/etc/profile.d/conda.sh
conda activate knomol-env

mkdir -p /vols/teaching/msc-projects/2024-2025/aanaidu/.dgl_cache
export DGL_HOME=/vols/teaching/msc-projects/2024-2025/aanaidu/.dgl_cache
mkdir -p $DGL_HOME
env | grep DGL_HOME

echo "Starting at $(date)"
python hyperparam_search_v4.py --dataset ClinTox --folds 5 --epoch 150
echo "Finished at $(date)"
