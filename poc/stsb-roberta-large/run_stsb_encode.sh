#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 25:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH -o stsb.log

export PATH=/usr/lib/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/lib/cuda-10.0/lib64:/usr/lib/matlab-9.7/bin/glnxa64

# Make conda available:
# Activate a conda environment:
source /INET/state-trolls/work/state-trolls/miniconda3/etc/profile.d/conda.sh
conda activate env3
export CUDA_VISIBLE_DEVICES=1-40
export GPU_DEVICE_ORDINAL=1-40
export TMPDIR=/INET/state-trolls/work/state-trolls/poc/tmp

# run the process to encode tweets
#srun --gres=gpu:4 python sts-trial.py &
python -u stsb_encode.py -i $1
# # code to print gpu stats while running the server
# for i in {0..300..2}
#   do 
#      nvidia-smi
#      sleep 15 
#  done
# wait 
