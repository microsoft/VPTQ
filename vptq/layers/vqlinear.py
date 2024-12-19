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
        debug=False,
        enable_proxy_error=True,
    ):
        super().__init__()

        # get init args
        self.init_args = {
            "in_features": in_features,
            "out_features": out_features,
            "vector_lens": vector_lens,
            "num_centroids": num_centroids,
            "num_res_centroids": num_res_centroids,
            "group_num": group_num,
            "group_size": group_size,
            "outlier_size": outlier_size,
            "enable_norm": enable_norm,
            "enable_perm": enable_perm,
            "bias": bias,
            "is_indice_packed": is_indice_packed,
        }

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # set configuration
        self.debug = debug
        self.enable_proxy_error = enable_proxy_error

        # to reduce index size and bypass nccl check
        self.indices_as_float = indices_as_float
        self.is_indice_packed = is_indice_packed

        # TODO: FIX magic number
        self.vector_len = vector_lens[1]
        self.num_centroids = num_centroids[1]
        self.num_res_centroids = num_res_centroids[1]

        self.group_num = group_num
        self.num_codebooks = self.group_num
        self.outlier_size = outlier_size

        assert vector_quant_dim in ["in", "out"]
        assert vector_quant_dim == "out"

        self.vector_quant_dim = vector_quant_dim
        if self.vector_quant_dim == "in":
            raise ValueError("Not implemented yet.")

        self.padding = (-self.out_features) % self.vector_len
        self.group_size = group_size
        self.transpose = True

        self.num_indices = (self.out_features + self.padding) // self.vector_len

        # set outliers
        self.outlier_padding = 0
        self.outlier_vector_len = -1
        self.outlier_num_centroids = -1
        self.outlier_indices = None
        self.outlier_centroids = None
        self.enable_outlier = False
        if vector_lens[0] > 1 and num_centroids[0] > 0:
            self.enable_outlier = True
            self.outlier_vector_len = vector_lens[0]
            self.outlier_num_centroids = num_centroids[0]
            self.outlier_num_res_centroids = num_res_centroids[0]
            self.outlier_padding = (-self.out_features) % self.outlier_vector_len
            self.ouliter_num_indices = (self.out_features + self.outlier_padding) // self.outlier_vector_len

            assert self.vector_quant_dim == "out", "Currently outlier only support vector quant on out_features"
            assert self.outlier_num_res_centroids == -1, "Currently do not support residual quant on outliers"
            self.outlier_centroids = nn.Embedding(
                1, self.outlier_num_centroids * self.outlier_vector_len, **factory_kwargs
            )

            # all index and perm are uint16 to avoid nccl and safetensor check
            # we view them as float16 or int16
            if self.indices_as_float:
                self.outlier_indices = Parameter(
                    torch.empty((1, self.ouliter_num_indices, self.outlier_size), dtype=torch.float16, device=device),
                    requires_grad=False,
                )
            else:
                self.outlier_indices = Parameter(
                    torch.empty((1, self.ouliter_num_indices, self.outlier_size), dtype=torch.int16, device=device),
                    requires_grad=False,
                )
    
        self.enable_residual = bool(self.num_res_centroids > 0)


        # set main centroids
        self.centroids = nn.Embedding(self.num_codebooks, self.num_centroids * self.vector_len, **factory_kwargs)

        # process norm
        self.enable_norm = enable_norm
        if self.enable_norm:
            if self.vector_quant_dim == "in":
                assert True, "Not implemented"
            else:
                self.weight_scale = Parameter(torch.empty(self.in_features, **factory_kwargs), requires_grad=True)
                self.weight_bias = Parameter(torch.empty(self.in_features, **factory_kwargs), requires_grad=True)

        # process permutation
        self.enable_perm = enable_perm
        if self.enable_perm:
            if self.vector_quant_dim == "in":
                assert True, "Not implemented"
            else:
                perm_dtype = torch.int16 if self.is_indice_packed else torch.int64
                self.perm = Parameter(
                    torch.arange(self.in_features, device=device, dtype=perm_dtype), requires_grad=False
                )

        # indices shape
        # self.num_indices in each codebook
        if self.vector_quant_dim == "in":
            assert True, "Not implemented"
        else:
            # packed indices
            if self.is_indice_packed is True:
                self.index_bits = int(math.log2(self.num_centroids))
                if self.enable_residual:
                    self.res_index_bits = int(math.log2(self.num_res_centroids))
                else:
                    self.res_index_bits = 0
                self.total_index_bits = self.index_bits + self.res_index_bits

                packed_groupsize = math.ceil(self.group_size * self.total_index_bits / 32)

                index_dtype = torch.int32 if self.is_indice_packed else torch.int16

                self.indices = Parameter(
                    torch.empty((self.num_codebooks, self.num_indices, packed_groupsize),
                                dtype=index_dtype,
                                device=device),
                    requires_grad=False,
                )
            else:
                # unpacked indices
                if self.indices_as_float:
                    self.indices = Parameter(
                        torch.empty((self.num_codebooks, self.num_indices, self.group_size),
                                    dtype=torch.float16,
                                    device=device),
                        requires_grad=False,
                    )
                else:
                    self.indices = Parameter(
                        torch.empty((self.num_codebooks, self.num_indices, self.group_size),
                                    dtype=torch.int16,
                                    device=device),
                        requires_grad=False,
                    )

        # set residual centroids and indices
        if self.enable_residual:
            self.res_centroids = nn.Embedding(
                self.num_codebooks, self.num_res_centroids * self.vector_len, **factory_kwargs
            )

            if self.is_indice_packed is False:
                if self.indices_as_float:
                    self.res_indices = Parameter(
                        torch.empty((self.num_codebooks, self.num_indices, self.group_size),
                                    dtype=torch.float16,
                                    device=device),
                        requires_grad=False,
                    )
                else:
                    self.res_indices = Parameter(
                        torch.empty((self.num_codebooks, self.num_indices, self.group_size),
                                    dtype=torch.int16,
                                    device=device),
                        requires_grad=False,
                    )
        else:
            self.res_centroids = self.register_parameter("res_centroids", None)
            self.res_indices = self.register_parameter("res_indices", None)

    # initialize parameters
    def init_parameters(
        self,
        centroids,
        indices,
        res_centroids=None,
        res_indices=None,
        weight_scale=None,
        weight_bias=None,
        weight=None,
        bias=None,
        perm=None,
        dtype=None,
    ):
        # step 1, handle outliers
        if self.enable_outlier:
            outlier_centroids = centroids[0].clone().detach().requires_grad_(True)
            outlier_centroids = outlier_centroids.reshape(1, self.outlier_num_centroids * self.outlier_vector_len)
            self.outlier_centroids.weight.data = outlier_centroids

            outlier_indices = indices[0]
            if self.indices_as_float:
                outlier_indices = (
                    outlier_indices.clone().detach().to(torch.uint16).view(torch.float16
                                                                          ).to(self.outlier_centroids.weight.device)
                )
            else:
                outlier_indices = (
                    outlier_indices.clone().detach().to(torch.uint16).view(torch.int16
                                                                          ).to(self.outlier_centroids.weight.device)
                )

            if len(outlier_indices.shape) == 2:
                outlier_indices = outlier_indices.unsqueeze(0)
            self.outlier_indices.data = outlier_indices

        # step 2, handle main centroids and indices
        _centroids = []
        keys = sorted(centroids.keys())
        for cidx in keys[1:]:  # main centroids start from 1
            _centroids.append(centroids[cidx])
        # (num_codebooks, num_centroids, vector_len)
        _centroids = torch.stack(_centroids, dim=0)

        _centroids = _centroids.reshape(self.num_codebooks, self.num_centroids * self.vector_len)
        self.centroids.weight.data = _centroids

        _indices = []
        keys = sorted(indices.keys())
        for cidx in keys[1:]:
            _indices.append(indices[cidx])
        _indices = torch.stack(_indices, dim=0)

        _indices = _indices.reshape(self.num_codebooks, self.num_indices, self.group_size)

        if self.indices_as_float:
            self.indices.data = _indices.to(torch.uint16).view(torch.float16).to(self.centroids.weight.device)
        else:
            self.indices.data = _indices.to(torch.uint16).view(torch.int16).to(self.centroids.weight.device)

        # step 3: handle residual
        if self.enable_residual:
            _res_centroids = []
            keys = sorted(res_centroids.keys())
            for cidx in keys[1:]:  # main centroids start from 1
                _res_centroids.append(res_centroids[cidx])
            _res_centroids = torch.stack(_res_centroids, dim=0)

            _res_centroids = _res_centroids.reshape(self.num_codebooks, self.num_res_centroids * self.vector_len)
            self.res_centroids.weight.data = _res_centroids


            # main indices
            _res_indices = []
            keys = sorted(res_indices.keys())
            for cidx in keys[1:]:  # main centroids start from 1
                _res_indices.append(res_indices[cidx])
            _res_indices = torch.stack(_res_indices, dim=0)

            _res_indices = _res_indices.reshape(self.num_codebooks, self.num_indices, self.group_size)

            if self.indices_as_float:
                self.res_indices.data = (
                    _res_indices.to(torch.uint16).view(torch.float16).to(self.res_centroids.weight.device)
                )
            else:
                self.res_indices.data = (
                    _res_indices.to(torch.uint16).view(torch.int16).to(self.res_centroids.weight.device)
                )

        if self.enable_norm:
            self.weight_scale.data = weight_scale.to(self.centroids.weight.device)
            self.weight_bias.data = weight_bias.to(self.centroids.weight.device)

        if self.enable_perm:
            self.perm.data = perm.to(self.centroids.weight.device)

    def set_centroids_grad(self, requires_grad):
        self.centroids.weight.requires_grad = requires_grad
        if self.enable_outlier:
            self.outlier_centroids.weight.requires_grad = requires_grad
        if self.enable_residual:
            self.res_centroids.weight.requires_grad = requires_grad

    def forward(self, x, W=None, H=None):
        # only for debug and layerwise finetuning
        if self.enable_proxy_error:
            return self.proxy_error_forward(W, H)
        else:
            outlier_centroids = self.outlier_centroids.weight if self.enable_outlier else None

            res_centroids = self.res_centroids.weight if self.res_centroids else None

            output = ops.dequant_gemm(
                x,
                self.bias,
                self.indices,
                self.centroids.weight,
                self.res_indices,
                res_centroids,
                self.outlier_indices,
                outlier_centroids,
                self.perm,
                self.weight_scale,
                self.weight_bias,
                self.num_codebooks,
                self.num_centroids,
                self.vector_len,
                self.num_res_centroids,
                self.outlier_num_centroids,
                self.outlier_vector_len,
                self.in_features,
                self.out_features
            )
            return output

    def proxy_error_forward(self, W, H):
        qweight = self.dequant()
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
        if self.vector_quant_dim == "in":
            raise AssertionError("self.vector_quant_dim == in")
        else:
            weights = F.pad(weights, (0, 0, 0, self.padding))
            weights = weights.T

        # (in, out) -> (in * out / vector, vector)
        vectors = weights.reshape(-1, self.vector_len).to("cuda")
        centroids = self.centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)

        indices = self._batched_indices(vectors, centroids)

        if self.enable_residual:
            res_vectors = vectors - centroids.squeeze(0)[indices.squeeze(0)]
            res_indices = self._batched_indices(
                res_vectors, self.res_centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)
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
        hatW = self.dequant()
        delta_w = hatW - W
        w_mean = torch.mean(W.T @ W * H)
        error_mean = torch.mean(delta_w.T @ delta_w * H)
        norm_error = error_mean / w_mean
        return error_mean, w_mean, norm_error
