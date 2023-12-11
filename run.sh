#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python main.py --mask False --base_path /scratch/connectome/stellasybae/ICASC- --experiment_name 231121_CheXPert_depth128 --layer_depth 128 # --dataset ADNI
# sbatch run_BWsimple.slurm
# sbatch run_BWexp.slurm
# sbatch run_BWexptemp.slurm

#rm -rf ./results/231201_CheXpert_bw_loss_exponential_and_temperature5_depth1/attention_map
CUDA_VISIBLE_DEVICES=6 python attention_map.py --wandb_mode offline --layer_depth 1 --experiment_name 231211_NIH_1_parallel_bw --test --dataset NIH --model_weight NIH_1_parallel_bw --batch-size 1

#CUDA_VISIBLE_DEVICES=1 python attention_map.py --wandb_mode offline --layer_depth 1 --bw_loss_setting exponential_and_temperature --experiment_name 231201_CheXpert_bw_loss_exponential_and_temperature5_depth1 --test --dataset CheXpert