# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cuml
import cupy
import numpy as np
import torch

from vptq.utils.reshape import reshape

def cupy_to_torch(cupy_array):
    return torch.utils.dlpack.from_dlpack(cupy_array.toDlpack())

@dataclass
class QuantizationArguments:
    vector_lens: List[int] = field(default_factory=lambda: [-1, 1])
    num_centroids: List[int] = field(default_factory=lambda: [-1, -1])
    num_res_centroids: List[int] = field(default_factory=lambda: [-1, -1])
    npercent: float = field(default=0)
    group_num: int = field(default=1)
    group_size: int = field(default=-1)
    kiter: int = field(default=100)
    ktol: float = field(default=1e-5)
    kseed: int = field(default=0)
    kmeans_mode: str = field(default=None)
    kmeans_alpha: float = field(default=0)
    enable_norm: bool = field(default=False)
    norm_dim: int = field(default=0)
    enable_perm: bool = field(default=False)


# N-percent outlier Vector Quantizator
# Partition data into N% outliers and (100-N)%.
class NPVectorQuantizer:

    def __init__(
        self,
        layer_name,
        logger,
        # vector quantization parameters
        vector_lens: Tuple[int, int],
        num_centroids: Tuple[int, int],
        num_res_centroids: Tuple[int, int],
        npercent: int,
        group_size: int,
        group_num: int,
        # kmeans parameters
        kmeans_mode: str = '',
        kmeans_seed: int = 0,
        # enable_transpose: bool = False,
        iter: int = 100,
        tol: float = 1e-5,
        # norm
        enable_norm: bool = False,
        norm_dim: int = 0,
        enable_perm: bool = False,
        debug: bool = False,
        # loaded_weights: dict = None,
    ):

        assert isinstance(num_centroids, (list, tuple))

        # self.enable_transpose = enable_transpose
        self.enable_transpose = True

        self.vector_lens = vector_lens
        self.num_centroids = num_centroids
        self.num_res_centroids = num_res_centroids
        self.npercent = npercent

        self.group_size = group_size
        self.group_num = group_num
        assert not ((self.group_size != -1) and (self.group_num != -1)), 'Can not set both group_size and group_num'
        self.iter = iter
        self.tol = tol
        self.kmeans_seed = kmeans_seed
        self.layer_name = layer_name

        # vector_len
        self.outlier_vector_len = self.vector_lens[0]
        self.vector_len = self.vector_lens[1]

        if self.outlier_vector_len > 0:
            self.enable_outlier = True
        else:
            self.enable_outlier = False

        # check kmeans_mode
        if kmeans_mode not in ['hessian', '']:
            raise ValueError(f'Not supported kmeans mode:{kmeans_mode}')

        self.kmeans_mode = kmeans_mode
        # self.kmeans_alpha = kmeans_alpha

        self.enable_norm = enable_norm
        self.norm_dim = norm_dim

        # centroids and indices
        self.centroids, self.indices = {}, {}
        # residual centroids and indices
        self.res_centroids, self.res_indices = {}, {}
        self.vector_norm = None

        # load checkpoint
        # self.loaded_weights = loaded_weights

        # reshape
        self.reshaper = {}
        self.res_reshaper = {}

        self.perm = None
        self.weight_scale = None
        self.weight_bias = None

        # debug
        self.debug = debug
        self.logger = logger

        # prefix layer name
        self.prefix_layer_name = 'model.layers.'

    def init_norm(self, weight):
        self.weight_scale = torch.std(weight, dim=self.norm_dim)
        self.weight_bias = torch.mean(weight, dim=self.norm_dim)

        if self.debug:
            self.logger.info(
                f'enabling norm dim {self.norm_dim}, '
                f'layer_name:{self.layer_name}, '
                f'scale:{self.weight_scale.shape}, '
                f'bias:{self.weight_bias.shape}'
            )

    # init permutation
    def init_perm(self, hessian, perm=None):
        if perm is not None:
            self.perm = perm
        else:
            self.perm = torch.argsort(torch.diag(hessian), descending=True)

    def get_centroid_cidx(self, index):
        if index < self.outlier_size:
            # if this index belongs to the first N%,
            # it should be quantized with the first codebook
            cidx = 0
        elif self.group_size != -1:
            # if use Product Quantization, find the corresponding codebook
            cidx = (index - self.outlier_size) // self.group_size + 1
        else:
            cidx = 1
        return cidx

    def get_group_setting(self, data):
        if self.enable_transpose:
            initial_outlier_size = int(math.ceil((self.npercent / 100) * data.shape[1]))
            remaining_columns = data.shape[1] - initial_outlier_size
            _pad = remaining_columns % self.group_num
            if _pad != 0:
                outlier_size = initial_outlier_size + _pad
            else:
                outlier_size = initial_outlier_size
            self.outlier_size = outlier_size
            if self.group_num != -1:
                group_size = (data.shape[1] - self.outlier_size) // self.group_num
                self.group_size = group_size
            else:
                assert True, 'only support transpose mode'
        else:
            assert True, 'only support transpose mode'

        return self.outlier_size, self.group_size, self.group_num

    def get_index_list(self, data):
        if self.group_size == -1 and self.group_num == -1:
            index_list = [[0, self.outlier_size], [self.outlier_size, None]]  # N% and (100-N)%
            return index_list

        # if setting group_size or num_goup, update self.group_size and self.group_num
        if self.group_size != -1:
            self.group_num = math.ceil((data.shape[1] - self.outlier_size) / self.group_size)
        elif self.group_num != -1:
            group_size = math.ceil((data.shape[1] - self.outlier_size) / self.group_num)
            if self.enable_transpose:
                self.group_size = group_size
            else:
                # group size should be multiple of vector_len
                self.group_size = math.ceil(group_size / self.vector_len) * self.vector_len
            assert self.group_num == math.ceil((data.shape[1] - self.outlier_size) / group_size)

        index_list = [[0, self.outlier_size]] + \
            [[self.outlier_size + t * self.group_size, self.outlier_size + (t+1) * self.group_size]
             for t in range(self.group_num)]

        if self.debug:
            self.logger.info(f'group_size: {self.group_size} '
                             f'number of groups: {self.group_num}')
        return index_list

    # k-means and quantize
    def init_centroids_indices(self, data, weights=None):
        '''
        do k-means on input data, update centroids and indices
        weights: do not support transpose, weight of each column (length should be data.shape[1])

        '''
        self.logger.info(
            f'data shape: {data.shape}, '
            f'weights shape: {weights.shape if weights is not None else None}'
        )

        quantized_data = []

        # Partition data into (1 + group_num) parts for Product Quantization:
        # the first N% columns: [0,outlier_size],
        # the last (100-N)% columns: [outlier_size, outlier_size + group_size],
        # [outlier_size + group_size, outlier_size + 2 * group_size], ...
        for idx, (i, j) in enumerate(self.get_index_list(data)):
            num_centroids = self.num_centroids[0] if idx == 0 else self.num_centroids[1]
            vector_len = self.outlier_vector_len if idx == 0 else self.vector_len
            train_data = data[:, i:j]
            train_weights = weights[:, i:j] if weights is not None else None

            if num_centroids == -1:  # Do not quantize, keep original data
                self.centroids[idx] = None
                self.indices[idx] = None
                self.logger.info(f'idx: {idx}, num_centroids: {num_centroids}, skip')
            else:
                self.reshaper[idx] = reshape(vector_len=vector_len, enable_transpose=self.enable_transpose)

                sub_vectors, padded_shape = self.reshaper[idx].matrix2vectors(train_data)
                vector_weights, _ = self.reshaper[idx].matrix2vectors(train_weights)

                # kmeans centroids from weight
                _kmeans = cuml.cluster.KMeans(
                    n_clusters=num_centroids, tol=self.tol, init='random', max_iter=self.iter, random_state=0, n_init=1
                )

                vector_weights = vector_weights.mean(dim=1) if vector_weights is not None else None

                # convert to numpy and float32 to avoid error
                sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()
                with cupy.cuda.Device(vector_weights.device.index):
                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)

                self.logger.info(f'cuml kmeans {_kmeans.n_iter_} iterations, error {_kmeans.inertia_}')

                self.centroids[idx] = torch.from_numpy(_kmeans.cluster_centers_).to(device=data.device)

                quant_data = self.centroids[idx][_kmeans.labels_]

                self.logger.info(f'idx: {idx}, quant_data shape: {quant_data.shape}')

                # reshape vectors to matrix
                quant_data = self.reshaper[idx].remove_padding(quant_data.reshape(padded_shape))

                self.logger.info(f'idx: {idx}, quant_data shape: {quant_data.shape}')
                quant_data.to(device=data.device)
                quantized_data.append(quant_data)

        quantized_data = torch.hstack(quantized_data)
        self.logger.info(f'quantized_data shape: {quantized_data.shape}')
        return quantized_data

    def quantize_vector(self, data, index):
        '''
        input data shape: if not transposed [-1,vector_len] else [nrows,1]
        index: The index of the first column of the input data in the entire weight matrix
        '''
        # Input check
        if self.enable_transpose:
            assert data.shape[1] == 1, 'only support quantize one column each time'
            data = data.T

        cidx = self.get_centroid_cidx(index)
        vector_len = self.outlier_vector_len if cidx == 0 else self.vector_len
        if self.centroids[cidx] is None:
            # keep original data for further quantization
            quantized_data = data
            self.indices[cidx] = None
        else:
            # matrix to vectors
            data, is_padded, pad_cols = self.reshaper[cidx].add_padding(data)

            if is_padded:
                assert is_padded == self.reshaper[cidx].is_padded \
                    and pad_cols == self.reshaper[cidx].pad_cols, \
                    f'cidx {cidx} index {index} pad_cols {pad_cols}' \
                    f'self.pad_cols {self.reshaper[cidx].pad_cols}' \
                    f'Error maybe caused by incorrect block_size settings'

            padded_shape = data.shape
            data = data.reshape(-1, vector_len)

            dist = torch.cdist(data.float(), self.centroids[cidx].float())

            indices = dist.argmin(dim=-1)
            quantized_data = self.centroids[cidx][indices]

            # save indices to self.indices
            # indices [out_feature / vector_len], 4096 / 8 = 512
            # self.logger.info(f'indices type: {type(indices), indices.shape}')
            if cidx not in self.indices or self.indices[cidx] is None:
                self.indices[cidx] = indices.unsqueeze(1).to(device=data.device)
            else:
                self.indices[cidx] = torch.hstack([self.indices[cidx], indices.unsqueeze(1).to(device=data.device)])
            # self.logger.info(f'self.indices[cidx]: {self.indices[cidx].shape}')

            # Reshape and remove padding if necessary
            quantized_data = quantized_data.reshape(padded_shape)
            quantized_data = self.reshaper[cidx].remove_padding(quantized_data)

        if self.enable_transpose:
            quantized_data = quantized_data.T

        return quantized_data

    def init_res_centroids_indices(self, data, weights=None):
        quantized_data = []
        for idx, (i, j) in enumerate(self.get_index_list(data)):
            vector_len = self.outlier_vector_len if idx == 0 else self.vector_len
            num_centroids = self.num_res_centroids[0] if idx == 0 else self.num_res_centroids[1]
            train_data = data[:, i:j]
            train_weights = weights[:, i:j] if weights is not None else None

            if num_centroids == -1:
                self.res_centroids[idx] = None
                self.res_indices[idx] = None
                self.logger.info(f'idx: {idx}, num_centroids: {num_centroids}, skip')
            else:
                self.res_reshaper[idx] = reshape(vector_len=vector_len, enable_transpose=self.enable_transpose)

                sub_vectors, padded_shape = self.res_reshaper[idx].matrix2vectors(train_data)
                vector_weights, _ = self.res_reshaper[idx].matrix2vectors(train_weights)

                # kmean
                _kmeans = cuml.cluster.KMeans(
                    n_clusters=num_centroids, tol=self.tol, init='random', max_iter=self.iter, random_state=0, n_init=1
                )

                self.logger.info(f'kmeans_mode: {self.kmeans_mode}, cuml kmeans, {num_centroids} clusters')
                # self.logger.info(sub_vectors.shape)
                sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()
                with cupy.cuda.Device(vector_weights.device.index):
                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)
                self.logger.info(f'cuml kmeans {_kmeans.n_iter_} iterations, error {_kmeans.inertia_}')

                self.res_centroids[idx] = torch.from_numpy(_kmeans.cluster_centers_).to(device=data.device)
                quant_data = self.res_centroids[idx][_kmeans.labels_]

                quant_data = self.res_reshaper[idx].remove_padding(quant_data.reshape(padded_shape))

                self.logger.info(f'idx: {idx}, res quant_data shape: {quant_data.shape}')

                quant_data.to(device=data.device)
                quantized_data.append(quant_data)

        # self.logger.info(f'{self.layername} kmeans error: ', torch.sum((data.double() - quant_data.double())**2).item())
        quantized_data = torch.hstack(quantized_data)
        return quant_data

    # residual quantize
    def quantize_residual_vector(self, data, index):
        '''
        input data shape: if not transposed [-1,vector_len] else [nrows,1]
        index: The index of the first column of the input data in the entire weight matrix
        '''
        # Input check
        if self.enable_transpose:
            assert data.shape[1] == 1, 'only support quantize one column each time'
            data = data.T

        cidx = self.get_centroid_cidx(index)
        vector_len = self.outlier_vector_len if cidx == 0 else self.vector_len
        if self.centroids[cidx] is None:
            quantized_data = data
            self.indices[cidx] = None
        else:
            data, is_padded, pad_cols = self.reshaper[cidx].add_padding(data)

            if is_padded:
                assert is_padded == self.reshaper[cidx].is_padded \
                    and pad_cols == self.reshaper[cidx].pad_cols, \
                    f'cidx {cidx} index {index} pad_cols {pad_cols}' \
                    f'self.pad_cols {self.reshaper[cidx].pad_cols[cidx]}' \
                    f'Error maybe caused by incorrect block_size settings'

            shape = data.shape
            data = data.reshape(-1, vector_len)

            # original centroid quantization
            dist = torch.cdist(data.float(), self.centroids[cidx].float())
            indices = dist.argmin(dim=-1)

            # self.logger.info(f'indices type: {type(indices), indices.shape}')

            if cidx not in self.indices or self.indices[cidx] is None:
                self.indices[cidx] = indices.unsqueeze(1)
            else:
                self.indices[cidx] = torch.hstack([self.indices[cidx], indices.unsqueeze(1)])
            # self.logger.info(f'self.indices[cidx]: {self.indices[cidx].shape}')

            quantized_data = self.centroids[cidx][indices]

            # residual quantization
            if self.res_centroids[cidx] is not None:
                residual_data = data - quantized_data
                dist = torch.cdist(residual_data.float(), self.res_centroids[cidx].float())
                res_indices = dist.argmin(dim=-1)

                # self.logger.info(f'residual indices type: {type(res_indices), res_indices.shape}')

                # self.logger.info(f'keys: {self.res_indices.keys()}, cidx: {cidx}')

                if cidx not in self.res_indices:
                    self.res_indices[cidx] = res_indices.unsqueeze(1)
                    # self.logger.info(f'type: {type(self.res_indices[cidx])}')
                    # self.logger.info(f'shape: {self.res_indices[cidx].shape}')
                else:
                    self.res_indices[cidx] = torch.hstack([self.res_indices[cidx], res_indices.unsqueeze(1)])
                    # self.logger.info(f'type: {type(self.res_indices[cidx])}')
                    # self.logger.info(f'type: {type(res_indices.unsqueeze(1))}, shape: {res_indices.unsqueeze(1).shape}')
                # self.logger.info(f'self.res_indices[cidx]: {self.res_indices[cidx].shape}')

                residual_quantized_data = \
                    self.res_centroids[cidx][res_indices]
                quantized_data = quantized_data + residual_quantized_data

            # Reshape and remove padding if necessary
            quantized_data = quantized_data.reshape(shape)
            quantized_data = self.reshaper[cidx].remove_padding(quantized_data)

        if self.enable_transpose:
            quantized_data = quantized_data.T

        return quantized_data

    def set_centroids(self, centroids, res_centroids=None):
        # check shape
        self.centroids = centroids.to(device=self.centroids.device)
        if res_centroids is not None:
            self.res_centroids = res_centroids.to(device=self.res_centroids.device)

    def clear_indices(self):
        self.indices = {0: None}
        self.res_indices = {0: None}
