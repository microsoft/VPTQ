from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from vptq.models.llama import eval_llama
from vptq.models.qwen import eval_qwen
from vptq.models.phi import eval_phi

from vptq.utils.data import get_data_loader
import vptq

import argparse

set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-65536-woft")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)

datasets = ['wikitext2', 'c4-new']
seqlens = [2048, 4096, 8192]
results = {}
model_name = model.config.model_type

for seqlen in seqlens:
    model.seqlen = seqlen
    for dataset in datasets:
        dataloader, testloader = get_data_loader(
            dataset, 
            seed=0, 
            model=model_name,
            seqlen=model.seqlen
        )
        
        print(f"Evaluating {dataset} with context length {seqlen}")
        
        if 'llama' in model_name.lower() or 'mistral' in model_name.lower():
            ppl = eval_llama(model, testloader, 'cuda')
        elif 'qwen' in model_name.lower():
            ppl = eval_qwen(model, testloader, 'cuda')
        elif 'phi' in model_name.lower():
            ppl = eval_phi(model, testloader, 'cuda')
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if f'ctx_{seqlen}' not in results:
            results[f'ctx_{seqlen}'] = {}
        results[f'ctx_{seqlen}'][dataset] = ppl
        print(f"PPL for {dataset}: {ppl}")
        