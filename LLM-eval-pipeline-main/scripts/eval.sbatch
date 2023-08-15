#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

model_id=phoenix-inst-chat-7b
batch_size=1

# MMCU, MMLU, GSM8K
benchmark_name=MMLU 
# few_shot, zero_shot, zero_shot_cot
setting=zero_shot


python eval.py \
    --model_id ${model_id} \
    --batch_size ${batch_size} \
    --benchmark_name ${benchmark_name} \
    --setting ${setting}
    
