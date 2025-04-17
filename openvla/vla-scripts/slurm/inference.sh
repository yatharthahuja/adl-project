#!/bin/bash
#SBATCH --job-name=64_hypertune
#SBATCH --output=/data/user_data/akirscht/slurm/out/%A_%a_ovla.out
#SBATCH --error=/data/user_data/akirscht/slurm/err/%A_%a_ovla.error
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=akirscht@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# Navigate to the directory containing your Python script
cd ~/openvla/vla-scripts/

# Run the Python script
python inference.py
