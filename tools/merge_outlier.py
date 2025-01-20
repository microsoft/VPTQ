import os
import torch
from transformers import AutoModelForCausalLM
f_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
bias_model = AutoModelForCausalLM.from_pretrained("/home/aiscuser/yangwang/VPTQ.dev/outputs/sphere-debug/bias_model_fake_quant")
# none_model = torch.load("/home/aiscuser/yangwang/VPTQ.dev/outputs/sphere-debug/2025-01-14-06-43-45/model.pt")
# bias_model = torch.load("/home/aiscuser/yangwang/VPTQ.dev/outputs/sphere-debug/2025-01-14-07-00-30/model.pt")


for layer_idx, (f_layer, bias_layer) in enumerate(zip(f_model.model.layers, bias_model.model.layers)):
    # List of weight tensors to process in each layer
    weight_pairs = [
        (f_layer.self_attn.q_proj, bias_layer.self_attn.q_proj),
        (f_layer.self_attn.k_proj, bias_layer.self_attn.k_proj),
        (f_layer.self_attn.v_proj, bias_layer.self_attn.v_proj),
        (f_layer.self_attn.o_proj, bias_layer.self_attn.o_proj),
        (f_layer.mlp.gate_proj, bias_layer.mlp.gate_proj),
        (f_layer.mlp.up_proj, bias_layer.mlp.up_proj),
        (f_layer.mlp.down_proj, bias_layer.mlp.down_proj),
    ]
    
    for f_op, bias_op in weight_pairs:
        # Get original weights from bias model
        bias_weights = bias_op.weight.detach()
        
        # Get weights from full precision model
        f_weights = f_op.weight.detach()
        
        # Calculate number of weights to replace (top 1%)
        n_elements = f_weights.numel()
        top_k = int(n_elements * 0.1)
        
        # Find indices of top 1% absolute values
        _, indices = torch.abs(f_weights).flatten().topk(top_k)
        
        # Create a mask for replacement
        mask = torch.zeros_like(f_weights).flatten()
        mask[indices] = 1
        mask = mask.reshape(f_weights.shape)
        
        # Replace values in bias_weights where mask is 1
        bias_weights = torch.where(mask.bool(), f_weights, bias_weights)
        
        # Requantize and update the bias model weights
        bias_op.weight.data = bias_weights

# Save the modified model
bias_model.save_pretrained("/home/aiscuser/yangwang/VPTQ.dev/outputs/sphere-debug/keep_max_0.1")