import argparse
import math
import os

import torch
import transformers
from tqdm import tqdm

import vptq
from vptq.utils.pack import dtype_convert, pack_index, unpack_index_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--orginal_model", type=str, default="")
parser.add_argument("--absorb_model", type=str, default="")
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.orginal_model)
model = vptq.AutoModelForCausalLM.from_pretrained(args.orginal_model, device_map='auto')


def process_vquantlinear(layer):
    if not hasattr(layer, 'enable_perm') or not layer.enable_perm:
        return

    print(f'layer.enable_perm: {layer.enable_perm}, perm dtype: {layer.perm.dtype}')
    # Get inverse permutation
    if layer.is_indice_packed:
        invert_perm = torch.argsort(layer.perm.view(torch.uint16).to(torch.int64))
    else:
        invert_perm = torch.argsort(layer.perm)

    # Rotate indices based on permutation
    if layer.is_indice_packed:
        index_bits = int(math.log2(layer.num_centroids))
        index_res_bits = int(math.log2(layer.num_res_centroids)) if layer.enable_residual else 0
        print(f'index_bits: {index_bits}, index_res_bits: {index_res_bits}')
        print(f'packed layer.indices shape: {layer.indices.shape}')
        print(f'layer.group_size: {layer.group_size}')

        # Unpack indices
        indices, res_indices = layer.unpack_index_tensor(
            pack_tensor=layer.indices,
            index_bits=index_bits,
            num_elements=layer.group_size,
            res_bits=index_res_bits,
            num_res_elements=layer.group_size,
            index_dtype=torch.uint16,
        )

        print(f'unpack indices shape: {indices.shape}, dtype: {indices.dtype}')
        print(f'unpack res_indices shape: {res_indices.shape}, dtype: {res_indices.dtype}')

        # Apply permutation
        indices = indices[..., invert_perm]
        if layer.enable_residual:
            res_indices = res_indices[..., invert_perm]

        indices = dtype_convert(indices, torch.int64, torch.uint16, torch.uint16)
        if layer.enable_residual:
            res_indices = dtype_convert(res_indices, torch.int64, torch.uint16, torch.uint16)

        print(f'perm indices shape: {indices.shape}')
        print(f'perm res_indices shape: {res_indices.shape}')

        # Pack indices back
        packed_indices = pack_index(
            indice=indices,
            index_bits=index_bits,
            res_indice=res_indices if layer.enable_residual else None,
            res_bits=index_res_bits if layer.enable_residual else 0,
            index_dtype=torch.uint16
        )

        # work around for packed indices shape
        print(f'packed_indices shape: {packed_indices.shape}')

        # Ensure packed shape matches original
        if packed_indices.shape != layer.indices.shape:
            raise ValueError(f"Packed shape {packed_indices.shape} doesn't match original shape {layer.indices.shape}")

        layer.indices.data = packed_indices
        print(f'repacked layer.indices shape: {layer.indices.shape}')
        print('-------')
    else:
        indices = layer.indices
        # indices = indices[..., invert_perm]
        layer.indices.data = indices

        if layer.enable_residual:
            res_indices = layer.res_indices
            # res_indices = res_indices[..., invert_perm]
            layer.res_indices.data = res_indices

    # Handle weight scale and bias if enable_norm is True
    if layer.enable_norm:
        if hasattr(layer, 'norm_dim') is False:
            layer.norm_dim = 0

    # # Disable permutation
    layer.enable_perm = False
    layer.perm = None


def process_model(model):
    # Process all VQuantLinear layers
    for name, module in model.named_modules():
        if isinstance(module, vptq.layers.vqlinear.VQuantLinear):
            process_vquantlinear(module)

    # update config
    for operator in model.config.quantization_config['config_for_layers'].keys():
        if model.config.quantization_config['config_for_layers'][operator]['enable_perm']:
            model.config.quantization_config['config_for_layers'][operator]['enable_perm'] = False


# Process the model
process_model(model)

# Save the processed model
output_path = args.absorb_model
if not os.path.exists(output_path):
    os.makedirs(output_path)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Model and tokenizer saved to {output_path}")
