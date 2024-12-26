# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from vptq import ops


class VQuantLinear(nn.Module):
    r"""Vector Quantized Linear Layer.
    
    Compute quantized matrix multiplication.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        vector_lens: Tuple containing the vector lengths for main vector
                     quantization and outlier vector quantization.
                     The first element represents the vector length of the
                     outlier component, while the second element represents
                     the vector length of the main component.
        num_centroids: Tuple containing the number of centroids for main vector
                       quantization and outlier vector quantization.
                       The first element represents the number of centroids for
                       the outlier component, while the second element
                       represents the number of centroids for the main
                       component.
        num_res_centroids: Tuple representing the number of residual centroids.
                           Currently, the second element indicates the number
                           of centroids for residual quantization methods.
                           If the value is set to -1, it indicates that
                           residual quantization is not used.
        group_num: Number of groups.
        group_size: Size of each group.
        outlier_size: Size of outliers.
        indices_as_float: Whether to use float indices.
        enable_norm: Whether to enable normalization.
        enable_perm: Whether to enable permutation.
        is_indice_packed: Whether indices are packed.
        bias: Whether to use bias.
        vector_quant_dim: Dimension to quantize vectors, in feature
                          or out feature.
        device: Device to use.
        dtype: Data type to use.
        enable_proxy_error: Whether to enable proxy error.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        vector_lens: Tuple[int, int],
        num_centroids: Tuple[int, int],
        num_res_centroids: Tuple[int, int],
        group_num: int,
        group_size: int,
        outlier_size: int,
        indices_as_float: bool,
        enable_norm: bool = False,
        enable_perm: bool = False,
        is_indice_packed: bool = False,
        bias: bool = False,
        vector_quant_dim: str = "out",
        device=None,
        dtype=None,
        enable_proxy_error=True
    ):
        super().__init__()
        if vector_quant_dim not in ["in", "out"]:
            raise ValueError("vector_quant_dim must be 'in' or 'out'.")

        if vector_quant_dim == "in":
            raise RuntimeError("Not implemented yet.")
        self.vector_quant_dim = vector_quant_dim

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(
                torch.empty(self.out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.enable_proxy_error = enable_proxy_error

        # === 1. process the main centroids
        # the second element of `vector_lens` and `num_centroids`
        # stores the information for the main vector quantization component.
        self.vector_len = vector_lens[1]
        self.num_centroids = num_centroids[1]

        self.padding = (-self.out_features) % self.vector_len
        self.num_indices = (self.out_features + self.padding) // self.vector_len

        # FIXME: these two fields are identical for legacy reasons
        self.group_num = group_num
        self.num_codebooks = self.group_num

        shape = (self.num_codebooks, self.num_centroids * self.vector_len)
        self.centroids = nn.Embedding(*shape, **factory_kwargs)

        self.indices_as_float = indices_as_float
        # all index and perm are uint16 to avoid nccl and safetensor check
        # we view them as float16 or int16
        index_type = torch.float16 if self.indices_as_float else torch.int16

        # === 2. process the outlier quantization component
        self.outlier_size = outlier_size
        self.outlier_padding = 0
        # the first element of `vector_lens` and `num_centroids`
        # stores the information for the outlier quantization component.
        self.outlier_vector_len = vector_lens[0]
        self.num_outlier_centroids = num_centroids[0]
        self.outlier_centroids = None

        self.outlier_indices = None
        self.ouliter_num_indices = 0

        self.outlier_num_res_centroids = num_res_centroids[0]
        self.enable_outlier = bool(
            self.outlier_vector_len > 1 and self.num_outlier_centroids > 0
        )

        if self.enable_outlier:
            if self.vector_quant_dim != "out":
                raise ValueError((
                    "Currently outlier is only supported for "
                    "vector quantization on out_features."
                ))
            if self.outlier_num_res_centroids != -1:
                raise ValueError((
                    "Current implementation does not support residual "
                    "quantization on outliers yet."
                ))

            self.outlier_padding = (
                -self.out_features
            ) % self.outlier_vector_len

            self.ouliter_num_indices = (
                self.out_features + self.outlier_padding
            ) // self.outlier_vector_len

            shape = (1, self.num_outlier_centroids * self.outlier_vector_len)
            self.outlier_centroids = nn.Embedding(*shape, **factory_kwargs)

            shape = (1, self.ouliter_num_indices, self.outlier_size)
            self.outlier_indices = Parameter(
                torch.empty(shape, dtype=index_type, device=device),
                requires_grad=False
            )

        # === 3. set centroids and indices for the residual quantization
        # to reduce index size and bypass nccl check
        self.is_indice_packed = is_indice_packed
        self.num_res_centroids = num_res_centroids[1]
        self.enable_residual = self.num_res_centroids > 0
        if self.enable_residual:
            self.res_indices = None
            self.res_centroids = nn.Embedding(
                self.num_codebooks, self.num_res_centroids * self.vector_len,
                **factory_kwargs
            )
            if self.is_indice_packed is False:
                # NOTE: When `is_indice_packed` is True, indices for the main
                # and residual quantization components are packed together.
                shape = (self.num_codebooks, self.num_indices, self.group_size)
                self.res_indices = Parameter(
                    torch.empty(shape, dtype=index_type, device=device),
                    requires_grad=False,
                )
        else:
            self.res_centroids = self.register_parameter("res_centroids", None)
            self.res_indices = self.register_parameter("res_indices", None)

        # === 4. process permutation
        self.enable_perm = enable_perm
        if self.enable_perm:
            perm_dtype = torch.int16 if self.is_indice_packed else torch.int64
            self.perm = Parameter(
                torch.arange(self.in_features, device=device, dtype=perm_dtype),
                requires_grad=False
            )

        # === 5. process normalization of the output
        self.enable_norm = enable_norm
        self.weight_bias = None
        self.weight_scale = None
        if self.enable_norm:
            self.weight_scale = Parameter(
                torch.empty(self.in_features, **factory_kwargs),
                requires_grad=True
            )
            self.weight_bias = Parameter(
                torch.empty(self.in_features, **factory_kwargs),
                requires_grad=True
            )

        # === 6. process packed indices
        self.group_size = group_size
        dtype = torch.int32 if self.is_indice_packed else torch.int16
        if self.is_indice_packed:
            self.index_bits = int(math.log2(self.num_centroids))

            self.res_index_bits = 0
            if self.enable_residual:
                self.res_index_bits = int(math.log2(self.num_res_centroids))
            self.total_index_bits = self.index_bits + self.res_index_bits

            packed_groupsize = math.ceil(
                self.group_size * self.total_index_bits / 32
            )

            shape = (self.num_codebooks, self.num_indices, packed_groupsize)
            self.indices = Parameter(
                torch.empty(shape, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            # unpacked indices
            shape = (self.num_codebooks, self.num_indices, self.group_size)
            self.indices = Parameter(
                torch.empty(shape, dtype=dtype, device=device),
                requires_grad=False
            )

    def init_parameters(
        self,
        centroids,
        indices,
        res_centroids=None,
        res_indices=None,
        weight_scale=None,
        weight_bias=None,
        perm=None
    ):
        # step 1, handle main centroids and indices
        _centroids = []
        keys = sorted(centroids.keys())
        for cidx in keys[1:]:  # main centroids start from 1
            _centroids.append(centroids[cidx])

        _centroids = torch.stack(_centroids, dim=0)
        _centroids = _centroids.reshape(
            self.num_codebooks, self.num_centroids * self.vector_len
        )
        self.centroids.weight.data = _centroids

        _indices = []
        keys = sorted(indices.keys())
        for cidx in keys[1:]:
            _indices.append(indices[cidx])
        _indices = torch.stack(_indices, dim=0)

        _indices = _indices.reshape(
            self.num_codebooks, self.num_indices, self.group_size
        )

        index_type = torch.float16 if self.indices_as_float else torch.uint16
        device = self.centroids.weight.device
        self.indices.data = _indices.to(torch.uint16
                                       ).view(index_type).to(device)

        # step 2, handle outliers
        if self.enable_outlier:
            outlier_centroids = centroids[0].clone().detach(
            ).requires_grad_(True)
            outlier_centroids = outlier_centroids.reshape(
                1, self.num_outlier_centroids * self.outlier_vector_len
            )
            self.outlier_centroids.weight.data = outlier_centroids

            outlier_indices = indices[0]
            device = self.outlier_centroids.weight.device

            dtype = torch.float16 if self.indices_as_float else torch.int16
            outlier_indices = outlier_indices.clone().detach().to(
                torch.uint16
            ).view(dtype).to(device)

            if len(outlier_indices.shape) == 2:
                outlier_indices = outlier_indices.unsqueeze(0)
            self.outlier_indices.data = outlier_indices

        # step 3: handle residual
        if self.enable_residual:
            _res_centroids = []
            keys = sorted(res_centroids.keys())
            for cidx in keys[1:]:  # main centroids start from 1
                _res_centroids.append(res_centroids[cidx])
            _res_centroids = torch.stack(_res_centroids, dim=0)

            _res_centroids = _res_centroids.reshape(
                self.num_codebooks, self.num_res_centroids * self.vector_len
            )
            self.res_centroids.weight.data = _res_centroids

            # main indices
            _res_indices = []
            keys = sorted(res_indices.keys())
            for cidx in keys[1:]:  # main centroids start from 1
                _res_indices.append(res_indices[cidx])
            _res_indices = torch.stack(_res_indices, dim=0)

            _res_indices = _res_indices.reshape(
                self.num_codebooks, self.num_indices, self.group_size
            )

            device = self.res_centroids.weight.device
            dtype = torch.float16 if self.indices_as_float else torch.uint16
            self.res_indices.data = _res_indices.to(torch.uint16
                                                   ).view(dtype).to(device)

        device = self.centroids.weight.device
        if self.enable_norm:
            self.weight_scale.data = weight_scale.to(device)
            self.weight_bias.data = weight_bias.to(device)

        if self.enable_perm:
            self.perm.data = perm.to(device)

    def set_centroids_grad(self, requires_grad):
        self.centroids.weight.requires_grad = requires_grad
        if self.enable_outlier:
            self.outlier_centroids.weight.requires_grad = requires_grad
        if self.enable_residual:
            self.res_centroids.weight.requires_grad = requires_grad

    def forward(self, x, W=None, H=None):
        if self.enable_proxy_error:
            # only for debug and layerwise finetuning
            return self.proxy_error_forward(W, H)

        outlier_centroids = None
        if self.enable_outlier:
            outlier_centroids = self.outlier_centroids.weight
        res_centroids = None
        if self.res_centroids is not None:
            res_centroids = self.res_centroids.weight

        output = ops.quant_gemm(
            x,
            bias=self.bias,
            indices=self.indices,
            centroids=self.centroids.weight,
            outlier_indices=self.outlier_indices,
            outlier_centroids=outlier_centroids,
            residual_indices=self.res_indices,
            residual_centroids=res_centroids,
            perm=self.perm,
            weight_scale=self.weight_scale,
            weight_bias=self.weight_bias,
            vector_len=self.vector_len,
            outlier_vector_len=self.outlier_vector_len,
            num_codebooks=self.num_codebooks,
            num_centroids=self.num_centroids,
            num_outlier_centroids=self.num_outlier_centroids,
            num_res_centroids=self.num_res_centroids,
            is_indice_packed=self.is_indice_packed,
            group_size=self.group_size,
            outlier_size=self.outlier_size,
            in_features=self.in_features,
            out_features=self.out_features,
            padding=self.padding,
            outlier_padding=self.outlier_padding,
            vector_quant_dim=self.vector_quant_dim
        )
        return output

    def proxy_error_forward(self, W, H):
        qweight = ops.dequant(
            indices=self.indices,
            centroids=self.centroids,
            outlier_indices=self.outlier_indices,
            outlier_centroids=self.outlier_centroids,
            res_indices=self.res_indices,
            res_centroids=self.res_centroids,
            perm=self.perm,
            weight_scale=self.weight_scale,
            weight_bias=self.weight_bias,
            is_indice_packed=self.is_indice_packed,
            enable_outlier=self.enable_outlier,
            enable_residual=self.enable_residual,
            enbale_perm=self.enable_perm,
            enable_norm=self.enable_norm,
            num_centroids=self.num_centroids,
            num_centroids_outlier=self.num_outlier_centroids,
            num_res_centroids=self.num_res_centroids,
            padding=self.padding,
            outlier_padding=self.outlier_padding,
            num_codebooks=self.num_codebooks,
            group_size=self.group_size,
            outlier_size=self.outlier_size,
            vector_len=self.vector_len,
            outlier_vector_len=self.outlier_vector_len,
            vector_quant_dim=self.vector_quant_dim
        )

        diff_weight = qweight - W
        proxy_error = diff_weight.T @ diff_weight * H
        return proxy_error

    def _batched_indices(self, vectors, centroids, batch_size=16384):
        vectors = vectors.cpu()
        centroids = centroids.to("cuda").float()
        n_vectors = vectors.shape[0]
        n_batches = (n_vectors + batch_size - 1) // batch_size
        indices = []
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_vectors)
            sub_vectors = vectors[start:end].to("cuda").float()
            dist_batch = torch.cdist(sub_vectors, centroids)
            indices_batch = torch.argmin(dist_batch, dim=-1)
            indices.append(indices_batch)
        return torch.hstack(indices)

    def _get_indices(self, vectors, centroids):
        centroids = centroids.to("cuda").float()
        sub_vectors = vectors
        dist_batch = torch.cdist(sub_vectors.float(), centroids)
        indices = torch.argmin(dist_batch, dim=-1)
        return indices

    # set indices by l2 distance
    def set_l2_indices(self, weights):
        weights = F.pad(weights, (0, 0, 0, self.padding))
        weights = weights.T

        # (in, out) -> (in * out / vector, vector)
        vectors = weights.reshape(-1, self.vector_len).to("cuda")
        centroids = self.centroids.weight.view(
            self.num_codebooks, self.num_centroids, self.vector_len
        )

        indices = self._batched_indices(vectors, centroids)

        if self.enable_residual:
            res_vectors = vectors - centroids.squeeze(0)[indices.squeeze(0)]
            res_indices = self._batched_indices(
                res_vectors,
                self.res_centroids.weight.view(
                    self.num_codebooks, self.num_centroids, self.vector_len
                )
            )
        # reshape indices and res_indices
        indices = indices.reshape(self.in_features, -1)
        indices = indices.T
        self.indices.data = indices.unsqueeze(0)

        if self.enable_residual:
            res_indices = res_indices.reshape(self.in_features, -1)
            res_indices = res_indices.T
        self.res_indices.data = res_indices.unsqueeze(0)

    # proxy error
    def get_error(self, W, H):
        hatW = ops.dequant(
            indices=self.indices,
            centroids=self.centroids,
            outlier_indices=self.outlier_indices,
            outlier_centroids=self.outlier_centroids,
            res_indices=self.res_indices,
            res_centroids=self.res_centroids,
            perm=self.perm,
            weight_scale=self.weight_scale,
            weight_bias=self.weight_bias,
            is_indice_packed=self.is_indice_packed,
            enable_outlier=self.enable_outlier,
            enable_residual=self.enable_residual,
            enbale_perm=self.enable_perm,
            enable_norm=self.enable_norm,
            num_centroids=self.num_centroids,
            num_centroids_outlier=self.num_outlier_centroids,
            num_res_centroids=self.num_res_centroids,
            padding=self.padding,
            outlier_padding=self.outlier_padding,
            num_codebooks=self.num_codebooks,
            group_size=self.group_size,
            outlier_size=self.outlier_size,
            vector_len=self.vector_len,
            outlier_vector_len=self.outlier_vector_len,
            vector_quant_dim=self.vector_quant_dim
        )
        delta_w = hatW - W
        w_mean = torch.mean(W.T @ W * H)
        error_mean = torch.mean(delta_w.T @ delta_w * H)
        norm_error = error_mean / w_mean
        return error_mean, w_mean, norm_error
