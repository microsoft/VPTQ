# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# import time
import math
from typing import Dict

from sentence_transformers.SentenceTransformer import SentenceTransformer
import torch


# from vptq.models.config import Config
# from vptq.models.llama import llama_eval
# from vptq.models.qwen import qwen_eval
# from vptq.models.data import get_loaders
# from vptq.models.config import Config
# from safetensors import safe_open
# from transformers import AutoTokenizer


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
        merged_indice = (res_indice.view(index_dtype).to(torch.uint64).view(torch.int64) << index_bits) | indice.view(
            index_dtype
        ).to(torch.uint64).view(torch.int64)
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
        res_indices = ((unpack_indice >> index_bits) & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
        res_indices = res_indices.squeeze()
    else:
        res_indices = None

    return indices, res_indices


def dtype_convert(data, from_dtype, to_dtype, as_type):
    data = data.view(from_dtype).to(to_dtype).view(as_type)
    return data


def convert_idx_dtype(model, from_dtype, to_dtype, as_type):
    print(f"converting model indices from {from_dtype} " f"to {to_dtype} as {as_type}")

    quant_config = {}
    for mod_name, sub_mod in model.named_modules():
        # print(f'mod_name: {mod_name}, sub_mod: {sub_mod}')
        if "VQuantLinear" in str(type(sub_mod)):
            sub_mod.cuda()
            # print(
            #     f'---debug---'
            #     f'index shape: {sub_mod.indices.shape}, '
            #     f'dtype: {sub_mod.indices.dtype}')

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

            # print(f'sub_mod.indices: {sub_mod.indices.shape}')

            # assert (sub_mod.fast_dequant() - sub_mod.dequant()).max().item() < 0.001
            sub_mod.cpu()
            quant_config[mod_name] = sub_mod.init_args
            quant_config[mod_name]["is_indice_packed"] = True

    if hasattr(model.config, "text_config"):
        model.config.text_config.quant_config = quant_config
    else:
        model.config.quant_config = quant_config
    return model


# # workaround for the issue that the bias field in the config is not updated correctly
def fix_tensor_in_config(model):
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    config_dict = config.to_dict()
    quant_config = config_dict["quant_config"]
    for layer_name, layer_config in quant_config.items():
        if isinstance(layer_config, Dict) and "bias" in layer_config.keys():
            if isinstance(layer_config["bias"], torch.Tensor):
                # print(f'Layer {layer_name} has a bias tensor')
                quant_config[layer_name]["bias"] = True  # Update bias to True
    config_dict["quant_config"] = quant_config
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
