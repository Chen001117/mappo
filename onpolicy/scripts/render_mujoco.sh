#!/bin/sh
env="MuJoCo"
scenario="Walker2d-v3"
num_agents=1
algo="rmappo"
exp="check"
seed=13

CUDA_VISIBLE_DEVICES=0 python3 render/render_mujoco.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
--num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
--use_render --episode_length 2048 --render_episodes 1 --use_ReLU --use_wandb --data_chunk_length 16 \
--model_dir "results/files"
