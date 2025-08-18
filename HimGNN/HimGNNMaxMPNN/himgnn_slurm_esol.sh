#!/bin/bash
#SBATCH -J himgnn_esol                 # Job name
#SBATCH --mail-user=aanaidu@stats.ox.ac.uk   # Your Stats email
#SBATCH --mail-type=BEGIN,END,FAIL     # Notifications
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10240M                    # Memory request

echo "Running on $(hostname)"

source /vols/teaching/msc-projects/2024-2025/aanaidu/yes/etc/profile.d/conda.sh
conda activate knomol-env


echo "Starting at $(date)"
python hyperparam_search.py --dataset ESOL --seed 1729 --folds 3 
echo "Finished at $(date)"
