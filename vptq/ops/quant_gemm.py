# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = [
    'dequant',
    'dequant_gemm',
]

import math

import torch
from torch.nn import functional as F

# isort: off
# we need to import the CUDA kernels after importing torch
__cuda_ops_installed = True
try:
    from vptq import cuda_ops
except ImportError:
    __cuda_ops_installed = False


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
    pad_size = (pack_tensor.shape[-1] *
                32) % (index_bits * num_elements + res_bits * num_res_elements)
    out = out.reshape(*pack_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*pack_tensor.shape[:-1], -1, total_bits)
    wf1 = torch.arange(0, total_bits,
                       1).to(pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)

    unpack_indice = out.to(torch.uint64).view(torch.int64)

    indices = (unpack_indice & ((1 << index_bits) - 1)).view(torch.uint64
                                                            ).to(torch.int64)

    if res_bits > 0:
        res_indices = ((unpack_indice >> index_bits) &
                       ((1 << index_bits) - 1)).view(torch.uint64
                                                    ).to(torch.int64)
    else:
        res_indices = None

    return indices, res_indices


def dequant(
    is_indice_packed: bool,
    num_centroids: int,
    num_res_centroids: int,
    enable_residual: bool,
    enable_outlier: bool,
    enable_perm: bool,
    enable_norm: bool,
    padding: int,
    outlier_padding: int,
    vector_quant_dim: str,
    group_size: int,
    outlier_size: int,
    num_codebooks: int,
    vector_len: int,
    outlier_vector_len: int,
    outlier_num_centroids: int,
    indices: torch.Tensor,
    res_indices: torch.Tensor,
    outlier_indices: torch.Tensor,
    centroids: torch.Tensor,
    res_centroids: torch.Tensor,
    outlier_centroids: torch.Tensor,
    perm: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
) -> torch.Tensor:
    if is_indice_packed:
        index_bits = math.ceil(math.log2(num_centroids))
        index_res_bits = 0

        if enable_residual:
            index_res_bits = math.ceil(math.log2(num_res_centroids))

        indices, res_indices = unpack_index_tensor(
            pack_tensor=indices,
            index_bits=index_bits,
            num_elements=group_size,
            res_bits=index_res_bits,
            num_res_elements=group_size,
            index_dtype=torch.uint16,
        )
    else:
        indices = indices.view(torch.uint16).to(torch.int64)
        if enable_residual:
            res_indices = res_indices.view(torch.uint16).to(torch.int64)

    indices = indices.unsqueeze(-1).expand(-1, -1, -1, vector_len)
    indices = indices.reshape(num_codebooks, -1, vector_len)

    selected_centroids = torch.gather(centroids, 1, indices)
    selected_centroids = selected_centroids.view(
        num_codebooks, -1, group_size, vector_len
    )

    selected_centroids = selected_centroids.permute(0, 1, 3, 2)
    qweight = selected_centroids.reshape(num_codebooks, -1, group_size)
    qweight = qweight.permute(1, 0, 2)
    qweight = qweight.reshape(-1, num_codebooks * group_size)

    if enable_residual:
        res_centroids = res_centroids.weight.view(
            num_codebooks, num_res_centroids, vector_len
        )
        res_indices = res_indices.unsqueeze(-1).expand(-1, -1, -1, vector_len)
        res_indices = res_indices.reshape(num_codebooks, -1, vector_len)

        selected_res_centroids = torch.gather(res_centroids, 1, res_indices)
        selected_res_centroids = selected_res_centroids.reshape(
            num_codebooks, -1, group_size, vector_len
        )
        selected_res_centroids = selected_res_centroids.permute(0, 1, 3, 2)

        qweight = qweight + (
            selected_res_centroids.reshape(num_codebooks, -1, group_size).
            permute(1, 0, 2).reshape(-1, num_codebooks * group_size)
        )

    # remove padding
    if padding > 0:
        if vector_quant_dim == "in":
            raise RuntimeError("Not implemented yet.")
        else:
            qweight = qweight[:-padding, :]

    if enable_outlier:
        outlier_centroids = outlier_centroids.weight.view(
            1, outlier_num_centroids, outlier_vector_len
        )

        outlier_indices = outlier_indices.view(torch.uint16).to(torch.int64)
        outlier_indices = outlier_indices.unsqueeze(-1).expand(
            -1, -1, -1, outlier_vector_len
        )

        outlier_indices = outlier_indices.reshape(1, -1, outlier_vector_len)

        selected_outlier_centroids = torch.gather(
            outlier_centroids, 1, outlier_indices
        )

        selected_outlier_centroids = selected_outlier_centroids.reshape(
            1, -1, outlier_size, outlier_vector_len
        )
        selected_outlier_centroids = selected_outlier_centroids.permute(
            0, 1, 3, 2
        )

        qweight_outlier = selected_outlier_centroids.reshape(-1, outlier_size)

        if outlier_padding > 0:
            if vector_quant_dim == "in":
                raise RuntimeError("Not implemented")
            else:
                qweight_outlier = qweight_outlier[:-outlier_padding,]
        qweight = torch.cat([qweight_outlier, qweight], dim=1)

    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        if vector_quant_dim == "in":
            raise RuntimeError("Not implemented")
        else:
            qweight = qweight[:, invert_perm]

    if enable_norm:
        qweight = qweight * weight_scale
        qweight = qweight + weight_bias

    return qweight


def dequant_gemm(
    x: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    centroids_: torch.Tensor,
    residual_indices: torch.Tensor,
    residual_centroids_: torch.Tensor,
    outlier_indices: torch.Tensor,
    outlier_centroids_: torch.Tensor,
    perm: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    num_codebooks: int,
    num_centroids: int,
    vector_len: int,
    num_res_centroids: int,
    outlier_num_centroids: int,
    outlier_vector_len: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Dequantize the input tensor and perform GEMM operation.
    
    Args:
        x: Tensor that has a shape of [batch_size, sequent_length, in_features]
        bias: Tensor
    """

    if indices.dtype != torch.int32:
        raise TypeError(
            "Indices that are not integers have not been implemented yet."
        )

    centroids = centroids_.view(num_codebooks, num_centroids, vector_len)

    residual_centroids = None
    if residual_centroids_ is not None:
        shape = (num_codebooks, num_res_centroids, vector_len)
        residual_centroids = residual_centroids_.weight.view(shape)

    outlier_centroids = None
    if outlier_centroids_ is not None:
        shape = (1, outlier_num_centroids, outlier_vector_len)
        outlier_centroids = outlier_centroids_.weight.view(shape)

    if x.numel() // x.shape[-1] < 3 and __cuda_ops_installed:
        out = cuda_ops.gemm(
            x,
            indices,
            centroids,
            residual_indices,
            residual_centroids,
            outlier_indices,
            outlier_centroids,
            perm,
            weight_scale,
            weight_bias,
            bias,
            vector_len,
            in_features,
            out_features,
        )
        return out
    else:
        if __cuda_ops_installed:
            weight = cuda_ops.dequant(
                indices,
                centroids,
                residual_indices,
                residual_centroids,
                outlier_indices,
                outlier_centroids,
                perm,
                weight_scale,
                weight_bias,
                vector_len,
                in_features,
                out_features,
            )
        else:
            weight = dequant(x)

        out = F.linear(x, weight, bias)
        return out
