#!/bin/sh

#SBATCH --job-name=llama3-2nodes8gpus
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhao.qian@ufl.edu # Where to send mail
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=200gb                   # Total memory limit
#SBATCH --partition=gpu               # Specify partition
#SBATCH --gres=gpu:a100:4             # Request 4 A100 GPUs per task
#SBATCH --account=ufhpc               # Sepcify account
#SBATCH --qos=ufhpc                   # Specify QoS setting
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load nlp/1.3

cd /data/ai/tutorial/Llama3_on_HPG/finetuning

# Running the torchrun command
torchrun --nproc_per_node=4 \
              finetuning.py --enable_fsdp --use_peft --peft_method lora \
              --model_name /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-hf \
              --output_dir /data/ai/tutorial/Llama3_on_HPG/finetuning/models/2nodes8gpus


