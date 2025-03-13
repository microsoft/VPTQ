# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import torch

import vptq


def ground_truth(
    x: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    centroids: torch.Tensor,
    residual_indices: torch.Tensor,
    residual_centroids: torch.Tensor,
    scale_weights: torch.Tensor,
    scale_bias: torch.Tensor,
    vector_len: int,
    out_features: int,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype

    batch_size, length, in_features = x.shape
    num_decoding = indices.shape[0]

    if num_decoding != residual_indices.shape[0]:
        raise ValueError(
            ("indices and residual_indices "
             "must have the same shape.")
        )
    if num_decoding != in_features * out_features // vector_len:
        raise ValueError((
            "indices must have the same shape as "
            "in_features * out_features // vector_len."
        ))

    # construct dequantized weights
    main_weights = torch.zeros(
        num_decoding, vector_len, dtype=dtype, device=device
    )
    residual_weights = torch.zeros(
        num_decoding, vector_len, dtype=dtype, device=device
    )
    residual_indices = residual_indices.to(torch.uint16)
    for i in range(num_decoding):
        idx = indices[i]
        main_weights[i, :] = centroids[0, idx, :]

        idx_residual = residual_indices[i]
        residual_weights[i, :] = residual_centroids[0, idx_residual, :]

    weights = main_weights + residual_weights
    weights = weights.reshape(in_features, out_features)
    weights = scale_weights * weights + scale_bias

    out = torch.zeros(
        batch_size, length, out_features, dtype=dtype, device=device
    )

    for i in range(batch_size):
        for j in range(length):
            vec = x[i, j, :]
            out[i, j, :] = vec @ weights
    out += bias
    return out


class TestQuantGemv(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)
        dtype = torch.bfloat16
        device = torch.device("cuda", 0)

        # gemv requires batch size * length < 16
        batch_size = 5
        length = 3

        self.in_features = 1024
        self.out_features = 2048

        self.num_codebooks = 1
        self.vector_length = 8

        self.num_centroids = 8192  # must be stored in uint16
        self.num_res_centroids = 256  # can be stored in uint8

        #====== generate data for unittest.  ======#
        # the activation tensor
        shape = (batch_size, length, self.in_features)
        self.x = torch.randn(*shape, device=device, dtype=dtype)

        # generate indices for unittest.
        num_indices = self.in_features * self.out_features // self.vector_length
        num_repeats = num_indices // self.num_centroids
        self.main_indices = torch.as_tensor(
            list(range(self.num_centroids)) * num_repeats,
            device=device,
            dtype=torch.uint16
        )

        num_repeats = num_indices // self.num_res_centroids
        self.residual_indices = torch.as_tensor(
            list(range(self.num_res_centroids)) * num_repeats,
            device=device,
            dtype=torch.uint8
        )

        num_repeats = (num_indices // self.num_res_centroids) * 2
        self.indices = torch.as_tensor(
            list(range(self.num_res_centroids)) * num_repeats,
            device=device,
            dtype=torch.uint16
        )

        shape = (self.num_codebooks, self.num_centroids, self.vector_length)
        self.centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (self.num_codebooks, self.num_res_centroids, self.vector_length)
        self.res_centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (1, 1, self.out_features)
        self.bias = torch.randn(*shape, device=device, dtype=dtype)

        # NOTE: In this test, the scale weights and bias are applied
        # to the in_features.
        shape = (self.in_features, 1)
        self.scale_weights = torch.randn(*shape, device=device, dtype=dtype)
        self.scale_bias = torch.randn(*shape, device=device, dtype=dtype)

    def test(self):
        out1 = vptq.ops.quant_gemv_v2(
            self.x,
            bias=self.bias,
            indices=self.indices,
            centroids=self.centroids,
            residual_indices=self.residual_indices,
            residual_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            num_codebooks=self.num_codebooks,
            num_centroids=self.num_centroids,
            num_residual_centroids=self.num_res_centroids,
            out_features=self.out_features
        )
        print(out1.shape)

        out2 = ground_truth(
            x=self.x,
            bias=self.bias,
            indices=self.main_indices,
            centroids=self.centroids,
            residual_indices=self.residual_indices,
            residual_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            out_features=self.out_features
        )
        print(out2.shape)


if __name__ == "__main__":
    unittest.main()
