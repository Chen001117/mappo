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
    CUDA_VISIBLE_DEVICES=1 python train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 8 --episode_length 256 --use_ReLU \
    --num_env_steps 100000000 --ppo_epoch 5 --entropy_coef 0.0 --use_eval --n_eval_rollout_threads 2 \
    --wandb_name "cwz19" --user_name "cwz19" --data_chunk_length 16 --lr 2e-5 --critic_lr 2e-5 --use_wandb
    # --model_dir "results/${env}/${scenario}/${algo}/check2/wandb/latest-run/files"
done
