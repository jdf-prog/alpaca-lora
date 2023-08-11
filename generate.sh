CUDA_VISIBLE_DEVICES=6 python generate.py \
    --load_8bit \
    --prompt_template "alpaca" \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_weights './lora-alpaca/meta-llama/Llama-2-7b-hf/new_data_qv_lora_layer_r8_alpha16_dropout0.05'