run_name="debug"
base_model="meta-llama/Llama-2-7b-hf"
output_dir="./lora-alpaca/${base_model}/${run_name}"
data_path="alpaca_data_min.json"
python finetune.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 1024 \
    --val_set_size 50 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length