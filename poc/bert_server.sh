#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH -o server.log

export PATH=/usr/lib/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/lib/cuda-10.0/lib64:/usr/lib/matlab-9.7/bin/glnxa64

# Make conda available:
# Activate a conda environment:
source /INET/state-trolls/work/state-trolls/miniconda3/etc/profile.d/conda.sh
conda activate env3
export CUDA_VISIBLE_DEVICES=1-40
export GPU_DEVICE_ORDINAL=1-40
export TMPDIR=/INET/state-trolls/work/state-trolls/poc/tmp

# logs the IP address of the server to be used by the Bert Client instantiation
ip a

# Run the server as a daemon process
srun --gres=gpu:4 bert-serving-start -model_dir ./models/uncased_L-12_H-768_A-12/ -max_batch_size 1024 -num_worker 4 -gpu_memory_fraction 1.0 -max_seq_len 300 -mask_cls_sep -device_map 0 1 2 3 &

# code to print gpu stats while running the server
for i in {0..300..2}
  do 
     nvidia-smi
     sleep 15 
 done
wait 


# bert-serving-start -model_dir ./models/uncased_L-12_H-768_A-12/ -num_worker 4 -gpu_memory_fraction 1.0 -max_seq_len 300 -mask_cls_sep -device_map 0 1 2 3
# srun --gres=gpu:1 -n1 --exclusive python nvdia.py &
# srun  --mem=16G bert-serving-start -model_dir ./models/uncased_L-12_H-768_A-12/ -num_worker 2 -gpu_memory_fraction 1.0 -max_seq_len 300 -mask_cls_sep -device_map 0 1 &
# srun python nvdia.py &
