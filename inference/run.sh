#!/bin/bash
#SBATCH -J leijingdi     # 作业名
#SBATCH -o logs/%x-%j.log   # stdout输出日志文件，%x是作业名，%j是job ID
#SBATCH -e logs/%x-%j.log   # stderr输出文件
#SBATCH -p AI4Phys   # 使用分区
#SBATCH -N 1                # 使用多少个节点
#SBATCH -n 16               # 申请的总cpu核心数，n卡程序核心数>=n
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=logs/error_%j.log      # 错误日志文件



python /mnt/hwfile/ai4chem/leijingdi/code/critic-v/inference/reason_critic.py \
    --reason_model_name  'qwen2-vl' \
    --reason_model_path  'Qwen/Qwen2-VL-7B-Instruct' \
    --critic_model_name  'qwen2-vl' \
    --critic_model_path  '/mnt/hwfile/ai4chem/CVPR/LLaMA-Factory/exported/critic-v-v1' \
    --dataset_name  'realWorldQA' \
    --dataset_path  '/mnt/hwfile/ai4chem/leijingdi/code/critic-v/inference/conversations.jsonl' \
    --world_size  1\
    --chat_round  2 
   