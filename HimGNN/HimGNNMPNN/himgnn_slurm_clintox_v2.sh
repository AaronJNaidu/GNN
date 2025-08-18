#!/bin/bash
#SBATCH -J himgnn_clintox                 # Job name
#SBATCH --mail-user=aanaidu@stats.ox.ac.uk   # Your Stats email
#SBATCH --mail-type=BEGIN,END,FAIL     # Notifications
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=102400M                    # Memory request

echo "Running on $(hostname)"

source /vols/teaching/msc-projects/2024-2025/aanaidu/yes/etc/profile.d/conda.sh
conda activate knomol-env


echo "Starting at $(date)"
python hyperparam_search_v2.py --dataset ClinTox --seed 1729 --folds 3 --epoch 150
echo "Finished at $(date)"
