#!/bin/bash
#SBATCH -J knomol_esol_edge_attention                 # Job name
#SBATCH --mail-user=aanaidu@stats.ox.ac.uk   # Your Stats email
#SBATCH --mail-type=BEGIN,END,FAIL     # Notifications
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10240M                    # Memory request

echo "Running on $(hostname)"

source /vols/teaching/msc-projects/2024-2025/aanaidu/yes/etc/profile.d/conda.sh
conda activate knomol_new


echo "Starting at $(date)"
python molnetdata.py --moldata ESOL --task reg 
echo "Processed data at $(date)"
python run_2.py search ESOL --task reg --numtasks 1 --seed 1729 --metric rmse --useedge --fold 5
echo "Finished at $(date)"
