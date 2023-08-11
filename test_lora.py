import os
import sys
import json

import fire
import gradio as gr
import torch
import transformers
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from string import Template
from collections import defaultdict

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import logging
from mt_metrics_eval.stats import Correlation
from typing import List

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
logging.warning("Using device: {}".format(device))

FINETUNE_INST = "You are evaluating the errors in a model-generated output for a(an) ${task} task."
FINETUNE_INPUT = """\
Task instruction: ${generation_instruction}
Source: ${input_context}
Model-generated Output: ${hypothesis_output}

Based on the given task instruction and source, identify the major and minor errors in this model-generated output.
Note that Major errors refer to actual errors that affects the task severely, and Minor errors refer to small imperfections, and purely subjective opinions about the output.
For each error you give in the response, please also elaborate the following information:
- error location (the words that are wrong in the output)
- error aspect it belongs to.
- explanation why it's an error, and the correction suggestions.
- severity of the error ("Major" or "Minor"). 
- reduction of score (between 0.5 and 5)

Your evaluation output in the json format:
"""

# FINETUNE_INST = ""
# FINETUNE_INPUT = """\
# Task instruction: ${generation_instruction}
# Source: ${input_context}
# Output: ${hypothesis_output}
# Based on the given task instruction and source, identify the major and minor errors in this output for a(an) ${task} task. Note that Major errors refer to actual errors that affects the task severely, and Minor errors refer to smaller imperfections, and purely subjective opinions about the output.
# For each error you give in the response, please also elaborate the following information:
# - error location (the words that are wrong in the output)
# - error aspect it belongs to.
# - explanation why it's an error, and the correction suggestions.
# - severity of the error ("Major" or "Minor"). 
# - reduction of score (Major for 5, Minor for 1)
# Your evaluation output in the json format:
# """



class MyCorrelation(Correlation):
    def __init__(self, num_sys:int, gold_scores:List[int], metric_scores:List[int]):
        # remove nan in metrics scores
        none_metric_scores_idxs = [idx for idx, x in enumerate(metric_scores) if x is None]
        logging.warning("Remove {} nan scores from {} scores".format(
            len(none_metric_scores_idxs),
            len(metric_scores)
        ))
        gold_scores = gold_scores.copy()
        # set gold scores to None if metric scores are None
        for idx in none_metric_scores_idxs[::-1]:
            gold_scores[idx] = None 
        super().__init__(num_sys, gold_scores, metric_scores)
        
def compute_correlation(output_file:str, human_score_name:str):
    if not human_score_name:
        logging.warning("No human score name provided, skip correlation computation.")
        return
    with open(output_file) as f:
        data = json.load(f)
    human_scores = [cand['scores'][human_score_name] for item in data for cand in item['candidates']]
    eval_scores = [cand['scores']['eval_xgptscore'] for item in data for cand in item['candidates']]
    cor = MyCorrelation(1, human_scores, eval_scores)
    logging.warning("Pearson correlation: {}".format(cor.Pearson()))
    logging.warning("Spearman correlation: {}".format(cor.Spearman()))
    logging.warning("Kendall correlation: {}".format(cor.Kendall()))


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    input_file: str = None,
    output_file: str = None,
    cor_human_score_name: str = None,
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    overwrite: bool = False,
    shard_size: int = None,
    shard_id: int = None,
):
    if os.path.exists(output_file) and not overwrite:
        logging.warning("Output file {} already exists, skip.".format(output_file))
        compute_correlation(output_file, cor_human_score_name)
        return
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side="left")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    # Load data
    with open(input_file) as f:
        data = json.load(f)
    if shard_size:
        total_shards = len(data) // shard_size + 1
        if shard_id >= total_shards:
            logging.warning("Shard id {} is out of range, skip.".format(shard_id))
            return
        data = data[shard_id * shard_size: (shard_id + 1) * shard_size]
        logging.warning("Shard {} of {} shards, load {} examples from {}".format(
            shard_id, total_shards, len(data), input_file))
    else:
        logging.warning("Load {} examples from {}".format(len(data), input_file))
    
    # Format data
    logging.warning("Formatting inputs...")

    formatted_data = []
    for item in data:
        for cand in item['candidates']:
            inst = Template(FINETUNE_INST).substitute(task=item['task'])
            input_ = Template(FINETUNE_INPUT).substitute(
                task=item['task'],
                generation_instruction=item['instruction'],
                input_context=item['input'],
                hypothesis_output=cand['text'],
            )
            formatted_data.append({
                "instruction": inst,
                "input": input_,
            })
    logging.warning("Formatted {} examples".format(len(formatted_data)))

    # Generate
    all_pure_outputs = []
    for i in tqdm(range(0, len(formatted_data), batch_size), desc="Generating"):
        batch_data = formatted_data[i:i+batch_size]
        batch_prompt = list(map(lambda x: prompter.generate_prompt(x['instruction'], x['input']), batch_data))
        batch_inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True)
        batch_input_ids = batch_inputs["input_ids"].to(device)
        batch_attention_mask = batch_inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
        )
        generate_params = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        batch_outputs = model.generate(**generate_params)
        batch_outputs_ids = batch_outputs.sequences
        batch_pure_outputs_ids = batch_outputs_ids[:, batch_input_ids.shape[1]:]
        batch_pure_outputs = tokenizer.batch_decode(batch_pure_outputs_ids, skip_special_tokens=True)

        all_pure_outputs.extend(batch_pure_outputs)
    logging.warning("Generated {} examples".format(len(all_pure_outputs)))

    def load_eval_output(eval_output):
        
        try:
            scores = defaultdict(int)
            json_eval_output = json.loads(eval_output)
            # for key in json_eval_output:
            #     for error in json_eval_output[key]:
            #         scores[key] -= error['score_reduction']
            #         scores['all'] -= error['score_reduction']
            for error in json_eval_output['errors'].values():
                scores[error["error_aspect"]] -= error['score_reduction']
                scores['all'] -= error['score_reduction']
            return json_eval_output, scores
        except Exception as e:
            logging.error("Error: {}".format(e))
            return eval_output, {'all': None}

        return json.loads(eval_output)
    idx = 0
    for item in data:
        for cand in item['candidates']:
            cand['eval_output'], cand['eval_scores'] = load_eval_output(all_pure_outputs[idx])
            cand['scores']['eval_xgptscore'] = cand['eval_scores']['all']
            idx += 1
    
    # Save data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        logging.warning("Save {} examples to {}".format(sum(len(x['candidates']) for x in data), output_file))
    
    # Compute correlation
    if cor_human_score_name:
        compute_correlation(output_file, cor_human_score_name)

        


    
if __name__ == "__main__":
    fire.Fire(main)