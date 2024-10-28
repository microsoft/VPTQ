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


class VQuantLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # [outlier vector length, vector length]
        # vector length
        vector_lens: Tuple[int, int],
        # centroids
        num_centroids: Tuple[int, int],
        num_res_centroids: Tuple[int, int],
        group_num: int,
        group_size: int,
        outlier_size: int,
        indices_as_float: bool,
        enable_norm: bool = False,
        enable_perm: bool = False,
        is_indice_packed: bool = False,
        # configuration
        bias: bool = False,
        vector_quant_dim: str = "out",
        # weight_scale=None,
        # weight_bias=None,
        device=None,
        dtype=None,
        debug=False,
        enable_proxy_error=True,
        # enable_outlier: bool = False,
        # outlier_vector_len: int = 0,
        # outlier_num_centroids: int = -1,
        # outlier_num_res_centroids: int = -1,
        # linear: nn.Linear,
        # vector quantization
        # res_centroids: torch.Tensor = None,
        # res_indices: torch.Tensor = None,
        # centroids: torch.Tensor,
        # indices: torch.Tensor,
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
        # # quantization
        # self.vector_lens = vector_lens
        # self.num_centroids = num_centroids
        # self.num_res_centroids = num_res_centroids

        # TODO: FIX magic number
        self.vector_len = vector_lens[1]
        self.num_centroids = num_centroids[1]
        self.num_res_centroids = num_res_centroids[1]

        # group_num = num_codebooks
        # self.group_num = group_num
        self.group_num = group_num
        self.num_codebooks = self.group_num
        self.outlier_size = outlier_size
        # print(f'group_num: {self.group_num}, group_size: {self.group_size}, '
        #       f'num_codebooks: {self.num_codebooks}')
        # the number of vector in group

        # set vector quantization parameters
        # current implementation only supports vector_len = 'out'
        assert vector_quant_dim in ["in", "out"]
        assert vector_quant_dim == "out"

        self.vector_quant_dim = vector_quant_dim
        # padding for vector quantization
        if self.vector_quant_dim == "in":
            assert True, "Not implemented"
        else:
            self.padding = (-self.out_features) % self.vector_len
            self.group_size = group_size
            self.transpose = True

        self.num_indices = (self.out_features + self.padding) // self.vector_len

        # set outliers
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

        else:
            self.enable_outlier = False

        if self.num_res_centroids > 0:
            self.enable_residual = True
        else:
            self.enable_residual = False

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

                # packed_groupsize = math.ceil(self.group_size *
                #                              (1 + (self.enable_residual is True))
                #                              * self.index_bits / 32
                #                              ) if self.is_indice_packed else self.group_size
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
            # if dtype is not None:
            #     self.outlier_centroids = self.outlier_centroids.to(dtype)

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

        # if dtype is not None:
        #     self.centroids = self.centroids.to(dtype)

        # print(
        #     f'self.centroids.weight.data: {self.centroids.weight.data.shape}')
        # main indices
        _indices = []
        keys = sorted(indices.keys())
        for cidx in keys[1:]:
            # print(f'indices[{cidx}]: {indices[cidx].shape}')
            _indices.append(indices[cidx])
        _indices = torch.stack(_indices, dim=0)
        # print(
        #     f'num_codebooks: {self.num_codebooks}, '
        #     f'num_indices: {self.num_indices}, '
        #     f'group_size: {self.group_size}')

        # print(f'_indices: {_indices.shape}')
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
            # (num_codebooks, num_centroids, vector_len)
            _res_centroids = torch.stack(_res_centroids, dim=0)

            _res_centroids = _res_centroids.reshape(self.num_codebooks, self.num_res_centroids * self.vector_len)
            self.res_centroids.weight.data = _res_centroids

            # if dtype is not None:
            #     self.res_centroids = self.res_centroids.to(dtype)

            # main indices
            _res_indices = []
            keys = sorted(res_indices.keys())
            for cidx in keys[1:]:  # main centroids start from 1
                # print(f'indices[{cidx}]: {res_indices[cidx].shape}')
                _res_indices.append(res_indices[cidx])
            _res_indices = torch.stack(_res_indices, dim=0)
            # print(
            #     f'num_codebooks: {self.num_codebooks}, '
            #     f'num_indices: {self.num_indices}, '
            #     f'group_size: {self.group_size}')

            # print(f'_res_indices: {_res_indices.shape}')
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

    # TODO: FIX
    def post_init(self):
        if not hasattr(self, "invert_perm"):
            self.invert_perm = (
                torch.argsort(self.perm.view(torch.uint16).to(torch.int64)).to(torch.uint16).view(torch.int16)
            )
        #     if self.indices.dtype != torch.int:
        #         self.short_indices = self.indices.view(
        #             torch.int16) if self.indices_as_fp16 else self.indices.short()
        #         self.short_res_indices = None
        #         if self.res_indices is not None:
        #             self.short_res_indices = self.res_indices.view(
        #                 torch.int16) if self.indices_as_fp16 else self.res_indices.short()
        #         self.short_outlier_indices = None
        #         if hasattr(self, "outlier_indices"):
        #             self.short_outlier_indices = self.outlier_indices.view(
        #                 torch.int16) if self.indices_as_fp16 else self.outlier_indices.short()
        #     self.short_perm = self.perm.short()

    # TODO: FIX
    def fast_gemv(self, x):
        try:
            from vptq import ops
        except ImportError:
            return None
        self.post_init()
        centroids = self.centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)
        res_centroids = (
            self.res_centroids.weight.view(self.num_codebooks, self.num_res_centroids, self.vector_len)
            if self.res_centroids is not None else None
        )
        outlier_centroids = (
            self.outlier_centroids.weight.view(1, self.outlier_num_centroids, self.outlier_vector_len)
            if hasattr(self, "outlier_centroids") else None
        )
        if self.indices.dtype == torch.int:
            indices = self.indices
            res_indices = self.res_indices if hasattr(self, "res_indices") else None
            outlier_indices = self.outlier_indices if hasattr(self, "outlier_indices") else None
        else:
            indices = self.short_indices
            res_indices = self.short_res_indices
            outlier_indices = self.short_outlier_indices
        out = ops.gemm(
            x,
            indices,
            centroids,
            res_indices,
            res_centroids,
            outlier_indices,
            outlier_centroids,
            self.perm,
            self.weight_scale,
            self.weight_bias,
            self.bias,
            self.vector_len,
            self.in_features,
            self.out_features,
        )
        return out

    def unpack_index_tensor(
        self,
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

        # indices = indices.squeeze()

        if res_bits > 0:
            res_indices = ((unpack_indice >> index_bits) & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
            # res_indices = res_indices.squeeze()
        else:
            res_indices = None

        return indices, res_indices

    def fast_dequant(self):
        try:
            from vptq import ops
        except ImportError:
            return None
        self.post_init()

        centroids = self.centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)
        res_centroids = (
            self.res_centroids.weight.view(self.num_codebooks, self.num_res_centroids, self.vector_len)
            if self.res_centroids is not None else None
        )
        outlier_centroids = (
            self.outlier_centroids.weight.view(1, self.outlier_num_centroids, self.outlier_vector_len)
            if hasattr(self, "outlier_centroids") else None
        )

        if self.is_indice_packed:
            indices = self.indices
            res_indices = self.res_indices if hasattr(self, "res_indices") else None
            outlier_indices = self.outlier_indices if hasattr(self, "outlier_indices") else None
        else:
            indices = self.short_indices
            res_indices = self.short_res_indices
            outlier_indices = self.short_outlier_indices

        output = ops.dequant(
            indices,
            centroids,
            res_indices,
            res_centroids,
            outlier_indices,
            outlier_centroids,
            self.invert_perm,
            self.weight_scale,
            self.weight_bias,
            self.vector_len,
            self.in_features,
            self.out_features,
        )
        return output

    def dequant(self):
        # if (output := self.fast_dequant()) is not None:
        #    return output

        centroids = self.centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)

        # print(f'indices fp16: {self.indices}')
        # print(f'indices uint16: {self.indices.view(torch.int16)}')
        if self.is_indice_packed:
            index_bits = math.ceil(math.log2(self.num_centroids))
            index_res_bits = math.ceil(math.log2(self.num_res_centroids)) if self.enable_residual else 0

            # print(f'self.indices shape: {self.indices.shape}')
            indices, res_indices = self.unpack_index_tensor(
                pack_tensor=self.indices,
                index_bits=index_bits,
                num_elements=self.in_features,
                res_bits=index_res_bits,
                num_res_elements=self.in_features,
                index_dtype=torch.uint16,
            )

            # print(f'indices: {indices.shape}')
            # if self.enable_residual:
            #     print(f'res_indices: {res_indices.shape}')

        else:
            indices = self.indices.view(torch.uint16).to(torch.int64)
            if self.enable_residual:
                res_indices = self.res_indices.view(torch.uint16).to(torch.int64)

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.vector_len)

        # print(f'2 indices: {indices.shape}')
        indices = indices.reshape(self.num_codebooks, -1, self.vector_len)
        # print(f'3 indices: {indices.shape}')
        # print(f'4 indices: {indices}')

        selected_centroids = torch.gather(centroids, 1, indices)
        # print(f'1 selected_centroids: {selected_centroids.shape}')
        # print(f'2 selected_centroids: {selected_centroids}')
        # selected_centroids = selected_centroids.view(
        #     self.num_codebooks, -1, self.in_features - len(self.outlier_idices), self.vector_len)
        selected_centroids = selected_centroids.view(self.num_codebooks, -1, self.group_size, self.vector_len)
        # print(f'3 selected_centroids: {selected_centroids.shape}')
        # print(f'4 selected_centroids: {selected_centroids}')
        selected_centroids = selected_centroids.permute(0, 1, 3, 2)
        # print(f'5 selected_centroids: {selected_centroids.shape}')
        # print(f'6 selected_centroids: {selected_centroids}')

        # print(self.num_codebooks, self.group_size)
        qweight = selected_centroids.reshape(self.num_codebooks, -1, self.group_size)
        qweight = qweight.permute(1, 0, 2)
        qweight = qweight.reshape(-1, self.num_codebooks * self.group_size)

        # print(f'qweight: {qweight.shape}')

        if self.enable_residual:
            res_centroids = self.res_centroids.weight.view(self.num_codebooks, self.num_res_centroids, self.vector_len)

            res_indices = res_indices.unsqueeze(-1).expand(-1, -1, -1, self.vector_len)

            res_indices = res_indices.reshape(self.num_codebooks, -1, self.vector_len)

            selected_res_centroids = torch.gather(res_centroids, 1, res_indices)

            selected_res_centroids = selected_res_centroids.reshape(
                self.num_codebooks, -1, self.group_size, self.vector_len
            )

            selected_res_centroids = selected_res_centroids.permute(0, 1, 3, 2)

            qweight = qweight + (
                selected_res_centroids.reshape(self.num_codebooks, -1, self.group_size
                                              ).permute(1, 0, 2).reshape(-1, self.num_codebooks * self.group_size)
            )

        # print(f'self.padding: {self.padding}')
        # print(f'self.out_features: {self.out_features}')
        # print(f'self.in_features: {self.in_features}')
        # # remove padding
        if self.padding > 0:
            if self.vector_quant_dim == "in":
                assert True, "Not implemented"
                qweight = qweight[:, :-self.padding]
            else:
                # if self.padding_indices > 0:
                #     qweight = qweight[:-self.padding, :-self.padding_indices]
                # else:
                qweight = qweight[:-self.padding, :]

        # print(f'qweight: {qweight.shape}')

        if self.enable_outlier:
            outlier_centroids = self.outlier_centroids.weight.view(
                1, self.outlier_num_centroids, self.outlier_vector_len
            )
            # outlier_centroids_shape = outlier_centroids.shape

            outlier_indices = self.outlier_indices.view(torch.uint16).to(torch.int64)

            outlier_indices = outlier_indices.unsqueeze(-1).expand(-1, -1, -1, self.outlier_vector_len)
            # print(f'0 outlier_indices: {outlier_indices.shape}')
            outlier_indices = outlier_indices.reshape(1, -1, self.outlier_vector_len)

            selected_outlier_centroids = torch.gather(outlier_centroids, 1, outlier_indices)
            # print(
            #     f'1 selected_outlier_centroids: {selected_outlier_centroids.shape}')
            selected_outlier_centroids = selected_outlier_centroids.reshape(
                1, -1, self.outlier_size, self.outlier_vector_len
            )
            # selected_outlier_centroids = selected_outlier_centroids.view(
            #     1, -1, len(self.outlier_indices), self.outlier_vector_len)
            # print(
            #     f'2 selected_outlier_centroids: {selected_outlier_centroids.shape}')
            selected_outlier_centroids = selected_outlier_centroids.permute(0, 1, 3, 2)
            # print(f'3 selected_outlier_centroids: {selected_outlier_centroids.shape}')

            qweight_outlier = selected_outlier_centroids.reshape(-1, self.outlier_size)

            if self.outlier_padding > 0:
                if self.vector_quant_dim == "in":
                    assert True, "Not implemented"
                else:
                    qweight_outlier = qweight_outlier[:-self.outlier_padding,]
            # print('qweight: ', qweight.shape)
            # print('qweight_outlier: ', qweight_outlier.shape)
            qweight = torch.cat([qweight_outlier, qweight], dim=1)
            # print('after concat: ', qweight.shape)

        if self.enable_perm:
            invert_perm = torch.argsort(self.perm.view(torch.uint16).to(torch.int64))
            if self.vector_quant_dim == "in":
                assert True, "Not implemented"
                # qweight = qweight[invert_perm, :]
            else:
                qweight = qweight[:, invert_perm]

        if self.enable_norm:
            qweight = qweight * self.weight_scale
            qweight = qweight + self.weight_bias

        return qweight

    def forward(self, x, W=None, H=None):
        # only for debug and layerwise finetuning
        if self.enable_proxy_error:
            return self.proxy_error_forward(W, H)
        else:
            if x.numel() // x.shape[-1] < 3 and (output := self.fast_gemv(x)) is not None:
                return output
            # debug
            # qweight = None
            qweight = self.fast_dequant()
            if qweight is None:
                qweight = self.dequant()
            return F.linear(x, qweight, self.bias)

    def proxy_error_forward(self, W, H):
        qweight = self.dequant()
        diff_weight = qweight - W
        proxy_error = diff_weight.T @ diff_weight * H
        # proxy_error = diff_weight.T @ diff_weight * H
        # proxy_error = diff_weight.T @ diff_weight * torch.diag(H)
        # proxy_error = diff_weight.T @ diff_weight * torch.eye(diff_weight.shape[1]).to(W.device)
        # loss = torch.mean(diff_weight.T @ diff_weight * H) +\
        #  torch.mean(diff_weight.T.detach() @ diff_weight.detach() * H)
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
            # vectors = vectors.cpu()
            # dist.append(dist_batch)
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

        # print(f'vectors: {vectors.shape}')
        # print(f'centroids: {centroids.shape}')

        indices = self._batched_indices(vectors, centroids)
        # indices = self._get_indices(vectors, centroids)

        # print(f'indices: {indices.shape}')
        # print(f'qvector: {centroids.squeeze(0)[indices.squeeze(0)].shape}')

        if self.enable_residual:
            res_vectors = vectors - centroids.squeeze(0)[indices.squeeze(0)]
            res_indices = self._batched_indices(
                res_vectors, self.res_centroids.weight.view(self.num_codebooks, self.num_centroids, self.vector_len)
            )
            # res_indices = self._get_indices(
            #     res_vectors, self.res_centroids.weight.view(
            #         self.num_codebooks, self.num_centroids, self.vector_len))

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
