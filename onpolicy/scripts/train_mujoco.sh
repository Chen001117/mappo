#!/bin/sh
env="MuJoCo"
scenario="Walker2d-v3"
num_agents=1
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train/train_mujoco.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 16 \
    --num_mini_batch 4 --episode_length 128 --use_ReLU --num_env_steps 1000000000 \
    --ppo_epoch 5 --entropy_coef 0. --use_eval --n_eval_rollout_threads 16 --hidden_size 64 \
    --critic_hidden_size 512 \
    --wandb_name "cwz19" --user_name "cwz19" --data_chunk_length 8 --lr 5e-5 --critic_lr 5e-5 \
    --gamma 0.93
    --model_dir "results/MuJoCo/Walker2d-v3/rmappo/check/wandb/run-20230410_012043-1zj3h8sy/files"
done
