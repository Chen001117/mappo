#!/bin/sh
env="MuJoCo"
scenario="Walker2d-v3"
num_agents=2
algo="rmappo"
exp="check"
seed=10

CUDA_VISIBLE_DEVICES=1 python3 render/render_mujoco.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
--num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
--use_render --episode_length 2048 --render_episodes 1 --use_ReLU --use_wandb --data_chunk_length 8 \
--hidden_size 64 --critic_hidden_size 256 \
--model_dir "results/files"
