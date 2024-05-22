#!/bin/sh

#SBATCH --job-name=llama3-4gpu
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhao.qian@ufl.edu # Where to send mail
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=200gb                    # Total memory limit
#SBATCH --partition=gpu               # Specify partition
#SBATCH --gres=gpu:a100:4             # Request 1 GPU per task
#SBATCH --account=ufhpc               # Sepcify account
#SBATCH --qos=ufhpc                   # Specify QoS setting
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load llama/3

# Important Note: Please specify the path to your own directory.
# Make sure to replace any placeholder paths with the actual directory path where you want to store or access your files.

cd /data/ai/tutorial/Llama3_on_HPG/finetuning

torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp --model_name /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-hf --use_peft --peft_method lora --output_dir /data/ai/tutorial/Llama3_on_HPG/finetuning/models/4gpu/peft