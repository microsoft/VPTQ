# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = [
    "dequant",
    "quant_gemm",
    "quant_gemv_v2",
]

import math
from typing import Optional

import torch
from torch.nn import functional as F

from vptq.utils.pack import unpack_index_tensor

__cuda_ops_installed = False

try:
    import vptq.libvptq as vptq_ops

    print("Successfully loaded VPTQ CUDA kernels.")
    __cuda_ops_installed = True
except Exception as error:
    print((
        f"{error}\n"
        "!!! Warning !!!: CUDA kernels are not found, "
        "please check CUDA and VPTQ installation."
    ))
    print((
        "!!! Warning !!!: Running on Torch implementations, "
        "which is extremely slow."
    ))


def dequant(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    outlier_indices: torch.Tensor,
    outlier_centroids: torch.Tensor,
    res_indices: torch.Tensor,
    res_centroids: torch.Tensor,
    perm: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    is_indice_packed: bool,
    enable_outlier: bool,
    enable_residual: bool,
    enable_perm: bool,
    enable_norm: bool,
    num_centroids: int,
    num_outlier_centroids: int,
    num_res_centroids: int,
    padding: int,
    outlier_padding: int,
    num_codebooks: int,
    group_size: int,
    outlier_size: int,
    vector_len: int,
    outlier_vector_len: int,
    vector_quant_dim: str = "out"
) -> torch.Tensor:
    if vector_quant_dim == "in":
        raise ValueError("Not implemented yet.")

    if is_indice_packed:
        index_bits = math.ceil(math.log2(num_centroids))
        index_res_bits = 0

        if enable_residual:
            index_res_bits = math.ceil(math.log2(num_res_centroids))

        indices, res_indices = unpack_index_tensor(
            packed_tensor=indices,
            index_bits=index_bits,
            num_elements=group_size,
            res_bits=index_res_bits,
            num_res_elements=group_size
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
        res_centroids = res_centroids.view(
            num_codebooks, num_res_centroids, vector_len
        )
        res_indices = res_indices.unsqueeze(-1).expand(-1, -1, -1, vector_len)
        res_indices = res_indices.reshape(num_codebooks, -1, vector_len)

        selected_res_centroids = torch.gather(res_centroids, 1, res_indices)
        res_centroids = selected_res_centroids.reshape(
            num_codebooks, -1, group_size, vector_len
        )
        res_centroids = res_centroids.permute(0, 1, 3, 2)
        res_centroids = res_centroids.reshape(num_codebooks, -1, group_size)
        res_centroids = res_centroids.permute(1, 0, 2).reshape(
            -1, num_codebooks * group_size
        )
        qweight = qweight + res_centroids

    if padding > 0:
        qweight = qweight[:-padding, :]  # remove padding

    if enable_outlier:
        outlier_centroids = outlier_centroids.view(
            1, num_outlier_centroids, outlier_vector_len
        )

        outlier_indices = outlier_indices.view(torch.uint16).to(torch.int64)
        outlier_indices = outlier_indices.unsqueeze(-1).expand(
            -1, -1, -1, outlier_vector_len
        )

        outlier_indices = outlier_indices.reshape(1, -1, outlier_vector_len)
        selected_outlier_centroids = torch.gather(
            outlier_centroids, 1, outlier_indices
        )

        outlier_centroids = selected_outlier_centroids.reshape(
            1, -1, outlier_size, outlier_vector_len
        )
        outlier_centroids = outlier_centroids.permute(0, 1, 3, 2)
        qweight_outlier = outlier_centroids.reshape(-1, outlier_size)

        if outlier_padding > 0:
            qweight_outlier = qweight_outlier[:-outlier_padding,]
        qweight = torch.cat([qweight_outlier, qweight], dim=1)

    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        qweight = qweight[:, invert_perm]

    if enable_norm:
        qweight = qweight * weight_scale + weight_bias

    return qweight


def quant_gemm(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    indices: torch.Tensor,
    centroids: torch.Tensor,
    outlier_indices: Optional[torch.Tensor],
    outlier_centroids: Optional[torch.Tensor],
    residual_indices: Optional[torch.Tensor],
    residual_centroids: Optional[torch.Tensor],
    perm: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    vector_len: int,
    outlier_vector_len: int,
    num_codebooks: int,
    num_centroids: int,
    num_outlier_centroids: int,
    num_res_centroids: int,
    is_indice_packed: bool,
    group_size: int,
    outlier_size: int,
    in_features: int,
    out_features: int,
    padding: int,
    outlier_padding: int,
    vector_quant_dim: str = "out"
) -> torch.Tensor:
    """Dequantize the input tensor and perform GEMM operation.
    """
    centroids_ = centroids.view(num_codebooks, num_centroids, vector_len)

    residual_centroids_ = None
    enable_residual = False
    if residual_centroids is not None:
        enable_residual = True
        shape = (num_codebooks, num_res_centroids, vector_len)
        residual_centroids_ = residual_centroids.view(shape)

    outlier_centroids_ = None
    enable_outlier = False
    if outlier_centroids is not None:
        enable_outlier = True
        shape = (1, num_outlier_centroids, outlier_vector_len)
        outlier_centroids_ = outlier_centroids.view(shape)

    enable_perm = perm is not None
    enable_norm = weight_scale is not None and weight_bias is not None

    invert_perm = None
    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        invert_perm = invert_perm.to(torch.uint16).view(torch.int16)

    if (x.numel() // x.shape[-1] < 3) and __cuda_ops_installed:
        out = vptq_ops.quant_gemv(
            x,
            indices,
            centroids_,
            residual_indices,
            residual_centroids_,
            outlier_indices,
            outlier_centroids_,
            perm,
            weight_scale,
            weight_bias,
            bias,
            in_features,
            out_features,
        )
        return out
    else:
        if __cuda_ops_installed:
            weight = vptq_ops.dequant(
                indices,
                centroids_,
                residual_indices,
                residual_centroids_,
                outlier_indices,
                outlier_centroids_,
                invert_perm,
                weight_scale,
                weight_bias,
                vector_len,
                in_features,
                out_features,
            )
        else:
            weight = dequant(
                indices=indices,
                centroids=centroids_,
                outlier_indices=outlier_indices,
                outlier_centroids=outlier_centroids_,
                res_indices=residual_indices,
                res_centroids=residual_centroids_,
                perm=perm,
                weight_scale=weight_scale,
                weight_bias=weight_bias,
                is_indice_packed=is_indice_packed,
                enable_outlier=enable_outlier,
                enable_residual=enable_residual,
                enable_perm=enable_perm,
                enable_norm=enable_norm,
                num_centroids=num_centroids,
                num_outlier_centroids=num_outlier_centroids,
                num_res_centroids=num_res_centroids,
                num_codebooks=num_codebooks,
                group_size=group_size,
                outlier_size=outlier_size,
                vector_len=vector_len,
                outlier_vector_len=outlier_vector_len,
                padding=padding,
                outlier_padding=outlier_padding,
                vector_quant_dim=vector_quant_dim
            )
        out = F.linear(x, weight, bias)
        return out
