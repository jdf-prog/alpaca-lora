#!/bin/bash
#SBATCH --job-name=alpaca_lora
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/%j.out

run_name="new_data_full_lora_layer_l1536_r8_alpha16_dropout0.05_lr2e-5"
base_model="meta-llama/Llama-2-13b-hf"
output_dir="./lora-alpaca/${base_model}/${run_name}"
# data_path="/home/dongfu/WorkSpace/ExplainableGPTScore/finetune_data/translation/train.json" # old data
data_path="/home/dongfu/WorkSpace/ExplainableGPTScore/data/wmt/train_data.wmt_mqm.distill_new_wmt_mqm.format.json" # new data
python finetune.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \
    --wandb_run_name $run_name \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --cutoff_len 1536 \
    --val_set_size 500 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte]' \
    --train_on_inputs \


# [q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte]