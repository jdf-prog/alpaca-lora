#!/bin/bash
#SBATCH --job-name=test_lora
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=./jobs/%j.out

# make ntasks = ngpus = num_shards

base_model="meta-llama/Llama-2-13b-hf"
lora_weights_name="new_data_full_lora_layer_l1536_r8_alpha16_dropout0.05_lr3e-4/checkpoint-179"
# lora_weights_name="new_data_qv_lora_layer_l1536_r8_alpha16_dropout0.05_lr2e-5"
lora_weights="./lora-alpaca/${base_model}/${lora_weights_name}"
input_file="/home/dongfu/WorkSpace/ExplainableGPTScore/data/wmt22/zh-en/eval_data.json"
output_file="./lora-alpaca/${base_model}/${lora_weights_name}/test_lora/eval_data.test_lora.json"
if [ ! -f "$lora_weights/adapter_model.bin" ]; then
    echo "lora_weights not found: ${lora_weights}/adapter_model.bin"
    exit 1
fi
if [ -d $(dirname ${output_file}) ]; then
    rm -rf $(dirname ${output_file})
    echo "Removed existing ${output_file}"
fi
mkdir -p $(dirname ${output_file})
batch_size=4
cor_human_score_name="mqm"
do_sharding="True"
get_json_len_py_cmd="import json; print(len(json.load(open(\"${input_file}\", 'r'))))"
if [ ${do_sharding} == "True" ]; then
    num_shards=4
    num_data=$(python -c "${get_json_len_py_cmd}")
    shard_size=$((num_data/num_shards+1))
    echo "num_data: ${num_data}"
    echo "shard_size: ${shard_size}"
    echo "num_shards: ${num_shards}"
    for i in $(seq 0 $((num_shards-1))); do
        shard_id=${i}
        shard_output_file="${output_file}.shard-${shard_id}"
        shard_output_log_file="${shard_output_file}.log"
        shard_shell_file="${shard_output_file}.sh"
        echo "shard_id: ${shard_id}"
        echo "shard_output_file: ${shard_output_file}"
        echo "shard_output_log_file: ${shard_output_log_file}"
        echo "shard_shell_file: ${shard_shell_file}"
        cmd="nvidia-smi && python test_lora.py \
            --base_model ${base_model} \
            --lora_weights ${lora_weights} \
            --prompt_template \"alpaca\" \
            --input_file ${input_file} \
            --output_file ${shard_output_file} \
            --batch_size ${batch_size} \
            --cor_human_score_name ${cor_human_score_name} \
            --shard_size ${shard_size} \
            --shard_id ${shard_id} "
        echo ${cmd} 1> ${shard_shell_file}
        
        srun --ntasks=1 --gpus-per-task=1 --cpus-per-gpu 1 --exclusive \
            bash ${shard_shell_file} > ${shard_output_log_file} 2>&1 &
    done
    jobs
    wait
    python agg_test_lora_files.py --output_file ${output_file}

else
    python test_lora.py \
    --base_model ${base_model} \
    --lora_weights ${lora_weights} \
    --prompt_template "alpaca" \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --batch_size ${batch_size} \
    --cor_human_score_name ${cor_human_score_name}

fi

