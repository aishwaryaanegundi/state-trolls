#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH --gpus-per-node=4
#SBATCH -o client.log

   
# Make conda available:
#eval "$(conda shell.bash hook)"
# Activate a conda environment:
source /INET/state-trolls/work/state-trolls/miniconda3/etc/profile.d/conda.sh
conda activate env3
export CUDA_VISIBLE_DEVICES=1-40
export GPU_DEVICE_ORDINAL=1-40
export TMPDIR=/INET/state-trolls/work/state-trolls/poc/tmp

python -u BertEncodingsForRedditData.py
