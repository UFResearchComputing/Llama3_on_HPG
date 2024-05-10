#!/bin/sh

#SBATCH --job-name=llama3-1gpu
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhao.qian@ufl.edu # Where to send mail
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=20gb                   # Total memory limit
#SBATCH --partition=gpu               # Specify partition
#SBATCH --gres=gpu:a100:1             # Request 1 GPU per task
#SBATCH --account=ufhpc               # Sepcify account
#SBATCH --qos=ufhpc                   # Specify QoS setting
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load nlp/1.3

cd /data/ai/tutorial/Llama3_on_HPG/finetuning

python finetuning.py --use_peft --peft_method lora --quantization \ 
       --dataset alpaca_dataset \
       --model_name /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-hf \
       --output_dir /data/ai/tutorial/Llama3_on_HPG/finetuning/models/1gpu/peft
