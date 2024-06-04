#!/bin/sh
env="MuJoCo"
scenario="Walker2d-v3"
num_agents=3
algo="rmappo"
exp="check"

seed=10

git add .
git commit -m 'exp'

CUDA_VISIBLE_DEVICES=0 python3 train/train_mujoco.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 16 \
    --num_mini_batch 4 --episode_length 1024 --use_ReLU --num_env_steps 1000000000 \
    --ppo_epoch 5 --entropy_coef 0 --hidden_size 64 --log_interval 1 \
    --critic_hidden_size 128 --data_chunk_length 8 --lr 0 --critic_lr 1e-4 \
    --wandb_name "cwz19" --user_name "cwz19" \
    --model_dir "results/MuJoCo/Walker2d-v3/rmappo/check/wandb/run-20240214_092405-tac15aus/files" 
