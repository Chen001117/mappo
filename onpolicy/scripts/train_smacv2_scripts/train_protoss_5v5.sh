env="StarCraft2v2"
map="10gen_protoss"
algo="rmappo"
units="5v5"

exp="800task"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
--seed 1 --units ${units} --n_training_threads 1 --n_rollout_threads 8 \
--num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 \
--use_value_active_masks --use_eval --eval_episodes 32 --use_linear_lr_decay
