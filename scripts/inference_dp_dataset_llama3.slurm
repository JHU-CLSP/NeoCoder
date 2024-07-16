#!/bin/bash -l
#SBATCH -A danielk80_gpu
#SBATCH --job-name=DPLLAMA3
#SBATCH --time=32:00:00
#SBATCH --output="test/log_llama3.txt"
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ylu130@jh.edu
#SBATCH --export=ALL

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load anaconda
conda activate creativity

# DISTRIBUTED_ARGS="--nproc_per_node $SLURM_NTASKS_PER_NODE \
#                   --nnodes $SLURM_NNODES \
#                   --master-addr $MASTER_ADDR \
#                   --master-port $MASTER_PORT"
BASE_PATH="/home/ylu130/workspace/unconventional-dataset"

cd $BASE_PATH
export PYTHONPATH=${BASE_PATH}

nvidia-smi -l 5 > test/gpu_usage.log &
PID=$!K

python steps/inference_dp.py \
          --dataset-path datasets/CodeForce/NeoCoder/gpt-4-1106-preview_diff=800_sample=200_dp=5.json \
          --model-name meta-llama/Meta-Llama-3-70B-Instruct \
          --dp-rounds 5 \
          --batch-size 1 \
          --output-dir datasets/CodeForce/inference \
          --overwrite
