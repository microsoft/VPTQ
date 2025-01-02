from transformers import AutoTokenizer, set_seed
import torch
from vptq.models.llama import eval_llama
from vptq.models.qwen import eval_qwen
from vptq.utils.data import get_data_loader
import vptq

set_seed(0)

model_path = "/home/aiscuser/yangwang/VPTQ.dev.rotate/outputs/Meta-Llama-3.1-8B-Instruct-test-in/2025-01-01-19-05-38/packed_model"
model = vptq.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").cpu()
model.eval()

datasets = ['wikitext2', 'c4-new']
seqlens = [2048, 4096, 8192]
results = {}

for seqlen in seqlens:
    model.seqlen = seqlen
    for dataset in datasets:
        dataloader, testloader = get_data_loader(
            dataset, 
            seed=0, 
            model=model_path,
            seqlen=model.seqlen
        )
        
        print(f"Evaluating {dataset} with context length {seqlen}")
        
        if 'llama' in model_path.lower() or 'mistral' in model_path.lower():
            ppl = eval_llama(model, testloader, 'cuda')
        elif 'qwen' in model_path.lower():
            ppl = eval_qwen(model, testloader, 'cuda')
            
        if f'ctx_{seqlen}' not in results:
            results[f'ctx_{seqlen}'] = {}
        results[f'ctx_{seqlen}'][dataset] = ppl
        print(f"PPL for {dataset}: {ppl}")
        