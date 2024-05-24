#!/bin/sh

#SBATCH --job-name=llama3-2nodes16gpus
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhao.qian@ufl.edu # Where to send mail
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --mem=800gb                   # Total memory limit
#SBATCH --partition=gpu               # Specify partition
#SBATCH --gres=gpu:a100:8             # Request 4 A100 GPUs per task
#SBATCH --account=ufhpc               # Sepcify account
#SBATCH --qos=ufhpc                   # Specify QoS setting
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load llama/3

# Important Note: Please specify the path to your own directory.
# Make sure to replace any placeholder paths with the actual directory path where you want to store or access your files.

cd /data/ai/tutorial/Llama2_on_HPG/finetuning || exit 1  # Ensure the directory exists

# Retrieve node list and IP address for rendezvous
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node /bin/hostname --ip-address)

# Check if torchrun is available and in the PATH
TORCHRUN_PATH=$(which torchrun)
if [ -z "$TORCHRUN_PATH" ]; then
  echo "Error: torchrun not found in PATH. Please ensure torchrun is installed and available in the PATH."
  exit 1
fi

# Ensure CUDA launch blocking is disabled
export CUDA_LAUNCH_BLOCKING=0

# Running the torchrun command
srun $TORCHRUN_PATH --nproc_per_node=8 --nnodes=2 --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:29500 finetuning.py --enable_fsdp --use_peft --peft_method lora \
         --model_name /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-hf \
         --output_dir /data/ai/tutorial/Llama3_on_HPG/finetuning/models/2nodes8gpus


