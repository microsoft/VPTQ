# merge two llama models
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_base = '/home/aiscuser/yangwang/vptq_abs_scan_fakequant/Meta-Llama-3.1-70B-Instruct-2024-12-11-01-44-42-12_model'
model_patch = '/home/aiscuser/yangwang/vptq_abs_scan_fakequant/Meta-Llama-3.1-70B-Instruct-2024-12-10-21-11-01-10_model'
model_merged = '/home/aiscuser/yangwang/merge_vptq_abs_scan_fakequant/Meta-Llama-3.1-70B-Instruct-2024-12-11-01-44-42-12_model+Meta-Llama-3.1-70B-Instruct-2024-12-10-21-11-01-10_model'

# model_base = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
# model_patch = '/home/aiscuser/yangwang/vptq_abs_scan_fakequant/Meta-Llama-3.1-70B-Instruct-2024-12-11-01-44-42-12_model'
# model_merged = '/home/aiscuser/yangwang/merge_vptq_abs_scan_fakequant/Meta-Llama-3.1-70B-Instruct-2024-12-11-01-44-42-12_+base'

model_base = AutoModelForCausalLM.from_pretrained(
    model_base,
    device_map='auto',
    torch_dtype=torch.bfloat16
)
model_patch = AutoModelForCausalLM.from_pretrained(
    model_patch,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

for name, param in model_base.named_parameters():
    print(f"scan {name}")
    if 'down_proj' in name:
        param.data.copy_(model_patch.get_parameter(name).data)
        print(f"merged {name}")

model_base.save_pretrained(
    model_merged,
    safe_serialization=True,
)

del model_patch
torch.cuda.empty_cache()
