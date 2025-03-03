#!/bin/bash


python inference/reason_critic.py \
    --reason_model_name  'qwen2-vl' \
    --reason_model_path  'Qwen/Qwen2-VL-7B-Instruct' \
    --critic_model_name  'qwen2-vl' \
    --critic_model_path  'Qwen/Qwen2-VL-7B-Instruct' \
    --dataset_name  'realWorldQA' \
    --dataset_path  '/mnt/hwfile/ai4chem/leijingdi/code/critic-v/inference/conversations.jsonl' \
    --world_size  1\
    --chat_round  2 
   