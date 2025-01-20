# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from typing import Dict

import torch
from sentence_transformers.SentenceTransformer import SentenceTransformer

# import time
import vptq


def pack_index(
    indice: torch.Tensor,
    index_bits: int,
    res_indice: torch.Tensor = None,
    res_bits: int = 0,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    total_bits = index_bits + res_bits
    assert total_bits <= 32, f"total index bits {total_bits} should be less than 32"
    assert as_dtype in [torch.int32], "as_dtype should be int32"

    # upcast the indice to uint64 to avoid overflow on signed bit
    if res_indice is not None:
        merged_indice = (res_indice.view(index_dtype).to(torch.uint64).view(torch.int64) <<
                         index_bits) | indice.view(index_dtype).to(torch.uint64).view(torch.int64)
    else:
        merged_indice = indice.view(index_dtype).to(torch.uint64).view(torch.int64)

    # merge the indice
    wf = torch.arange(0, total_bits).to(merged_indice.device).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(merged_indice.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(*merged_indice.shape[:-1], -1)
    paded_bits = (32 - out.reshape(*merged_indice.shape[:-1], -1).shape[-1] % 32) % 32
    out = torch.nn.functional.pad(
        out,
        (0, paded_bits),
        value=0,
        mode="constant",
    ).reshape(*merged_indice.shape[:-1], -1, 32)
    wf1 = torch.arange(0, 32, 1).to(merged_indice.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1)
    out = out.sum(dim=-1).to(torch.uint32).view(as_dtype)

    unpack_indices = unpack_index_tensor(
        out,
        index_bits,
        indice.shape[-1],
        res_bits,
        res_indice.shape[-1] if res_indice is not None else 0,
        index_dtype=index_dtype,
        as_dtype=as_dtype,
    )
    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_indices[0],
    )

    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_index_tensor(
            out,
            index_bits,
            indice.shape[-1],
            res_bits,
            res_indice.shape[-1] if res_indice is not None else 0,
            index_dtype=index_dtype,
            as_dtype=as_dtype,
        )[0],
    )
    if res_indice is not None:
        assert torch.allclose(
            res_indice.view(index_dtype).to(torch.int64),
            unpack_index_tensor(
                out,
                index_bits,
                indice.shape[-1],
                res_bits,
                res_indice.shape[-1] if res_indice is not None else 0,
                index_dtype=index_dtype,
                as_dtype=as_dtype,
            )[1],
        )
    return out


def unpack_index_tensor(
    pack_tensor: torch.Tensor,
    index_bits: int,
    num_elements: int,
    res_bits: int = 0,
    num_res_elements: int = 0,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    total_bits = index_bits + res_bits
    wf = torch.arange(0, 32, 1).to(pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, -1), wf)
    torch.bitwise_and(out, 1, out=out)
    pad_size = (pack_tensor.shape[-1] * 32) % (index_bits * num_elements + res_bits * num_res_elements)
    out = out.reshape(*pack_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*pack_tensor.shape[:-1], -1, total_bits)
    wf1 = torch.arange(0, total_bits, 1).to(pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)

    unpack_indice = out.to(torch.uint64).view(torch.int64)

    indices = (unpack_indice & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
    indices = indices.squeeze()

    if res_bits > 0:
        res_indices = ((unpack_indice >> index_bits) & ((1 << res_bits) - 1)).view(torch.uint64).to(torch.int64)
        res_indices = res_indices.squeeze()
    else:
        res_indices = None

    return indices, res_indices


def dtype_convert(data, from_dtype, to_dtype, as_type):
    data = data.view(from_dtype).to(to_dtype).view(as_type)
    return data


def convert_idx_dtype(model, from_dtype, to_dtype, as_type):
    print(f"converting model indices from {from_dtype} "
          f"to {to_dtype} as {as_type}")

    quantization_config = {}
    quantization_config["quant_method"] = "vptq"
    quantization_config["config_for_layers"] = {}

    for mod_name, sub_mod in model.named_modules():
        # print(f'mod_name: {mod_name}, sub_mod: {sub_mod}')
        if "VQuantLinear" in str(type(sub_mod)):
            sub_mod.cuda()

            if sub_mod.indices.dtype == torch.int64:
                sub_mod.indices.data = dtype_convert(
                    sub_mod.indices.data, sub_mod.indices.data.dtype, to_dtype, as_type
                )
            else:
                sub_mod.indices.data = dtype_convert(sub_mod.indices.data, from_dtype, to_dtype, as_type)

            if hasattr(sub_mod, "res_indices") and sub_mod.res_indices is not None:
                if sub_mod.res_indices.dtype == torch.int64:
                    sub_mod.res_indices.data = dtype_convert(
                        sub_mod.res_indices.data, sub_mod.res_indices.data.dtype, to_dtype, as_type
                    )
                else:
                    sub_mod.res_indices.data = dtype_convert(sub_mod.res_indices.data, from_dtype, to_dtype, as_type)

            if hasattr(sub_mod, "outlier_indices") and sub_mod.outlier_indices is not None:
                if sub_mod.outlier_indices.dtype == torch.int64:
                    sub_mod.outlier_indices.data = dtype_convert(
                        sub_mod.outlier_indices.data, sub_mod.outlier_indices.data.dtype, to_dtype, as_type
                    )
                else:
                    sub_mod.outlier_indices.data = dtype_convert(
                        sub_mod.outlier_indices.data, from_dtype, to_dtype, as_type
                    )
            # if sub_mod has perm
            if hasattr(sub_mod, 'perm'):
                if sub_mod.perm.dtype == torch.int64:
                    sub_mod.perm.data = dtype_convert(sub_mod.perm.data, sub_mod.perm.data.dtype, to_dtype, as_type)
                else:
                    sub_mod.perm.data = dtype_convert(sub_mod.perm.data, from_dtype, to_dtype, as_type)

            sub_mod.indices.data = pack_index(
                indice=sub_mod.indices,
                index_bits=int(math.log2(sub_mod.num_centroids)),
                res_indice=sub_mod.res_indices,
                res_bits=int(math.log2(sub_mod.num_res_centroids)) if sub_mod.res_indices is not None else 0,
                index_dtype=to_dtype,
            ).data

            sub_mod.res_indices = None

            sub_mod.cpu()
            quantization_config["config_for_layers"][mod_name] = sub_mod.init_args
            quantization_config["config_for_layers"][mod_name]["is_indice_packed"] = True

    if hasattr(model.config, "text_config"):
        model.config.text_config.quantization_config = quantization_config
    else:
        model.config.quantization_config = quantization_config
    return model


# # workaround for the issue that the bias field in the config is not updated correctly
def fix_tensor_in_config(model):
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    config_dict = config.to_dict()
    quantization_config = config_dict["quantization_config"]
    for layer_name, layer_config in quantization_config.items():
        if isinstance(layer_config, Dict) and "bias" in layer_config.keys():
            if isinstance(layer_config["bias"], torch.Tensor):
                # print(f'Layer {layer_name} has a bias tensor')
                quantization_config[layer_name]["bias"] = True  # Update bias to True
    config_dict["quantization_config"] = quantization_config
    if hasattr(model.config, "text_config"):
        model.config.text_config.update(config_dict)
    else:
        model.config.update(config_dict)
    return model


def pack_model(qmodel, from_type, to_type, as_type):
    if isinstance(qmodel, SentenceTransformer):
        st_model = qmodel

        qmodel = qmodel._modules["0"]._modules["auto_model"]
        model = convert_idx_dtype(qmodel, from_type, to_type, as_type)
        model = fix_tensor_in_config(model)
        st_model._modules["0"]._modules["auto_model"] = model

        return st_model
    model = convert_idx_dtype(qmodel, from_type, to_type, as_type)
    model = fix_tensor_in_config(model)
    return model


def absorb_perm_layer(layer):
    if not hasattr(layer, 'enable_perm') or not layer.enable_perm:
        return False

    if layer.group_num > 1:
        print(f'{layer.name} has {layer.group_num} groups, skipping perm absorption')
        return False

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

    return True


def absorb_perm(model):
    absorbed_perm = False
    # Process all VQuantLinear layers
    for name, module in model.named_modules():
        if isinstance(module, vptq.layers.vqlinear.VQuantLinear):
            absorbed_perm = absorb_perm_layer(module)

    # update config
    if absorbed_perm:
        for operator in model.config.quantization_config['config_for_layers'].keys():
            if model.config.quantization_config['config_for_layers'][operator]['enable_perm']:
                model.config.quantization_config['config_for_layers'][operator]['enable_perm'] = False

    return model
