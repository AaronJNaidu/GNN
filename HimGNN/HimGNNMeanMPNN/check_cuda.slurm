#!/bin/bash
#SBATCH --job-name=check-cuda
#SBATCH --output=check_cuda_output.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --mem=4G

echo "Running on $(hostname)"
echo "Starting at $(date)"

# Activate your Conda environment
source /vols/teaching/msc-projects/2024-2025/aanaidu/yes/miniconda3/etc/profile.d/conda.sh
conda activate knomol-env-backup

# Print PyTorch and CUDA version
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA version:', torch.version.cuda)"

echo "Finished at $(date)"
