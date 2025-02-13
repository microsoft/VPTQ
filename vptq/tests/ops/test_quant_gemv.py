# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import torch

import vptq


class TestQuantGemv(unittest.TestCase):

    def setUp(self):
        dtype = torch.bfloat16
        # dtype = torch.float16

        device = torch.device("cuda", 0)

        # gemv requires batch size * length < 16
        batch_size = 5
        length = 3

        self.in_features = 4096
        self.out_features = 14336
        self.num_codebooks = 1
        self.vector_length = 8
        self.num_centroids = 8192
        self.num_res_centroids = 256

        num_padding = (-self.out_features) % self.vector_length
        num_vecs = (self.out_features + num_padding) // self.vector_length

        torch.manual_seed(1234)

        # the activation tensor
        shape = (batch_size, length, self.in_features)
        self.x = torch.randn(*shape, device=device, dtype=dtype)

        # the quantized weight tensor
        shape = (self.num_codebooks, num_vecs, self.in_features)
        self.indices = torch.randint(
            low=0,
            high=self.num_centroids - 1,
            size=shape,
            device=device,
            dtype=torch.int32
        )

        shape = (self.num_codebooks, self.num_centroids, self.vector_length)
        self.centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (self.num_codebooks, self.num_res_centroids, self.vector_length)
        self.res_centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (1, self.out_features)
        self.bias = torch.randn(*shape, device=device, dtype=dtype)

        # NOTE: In this test, the scale weights and bias are applied
        # to the in_features.
        shape = (1, self.in_features)
        self.scale_weights = torch.randn(*shape, device=device, dtype=dtype)
        self.scale_bias = torch.randn(*shape, device=device, dtype=dtype)

    def test(self):
        out = vptq.ops.quant_gemv_v2(
            self.x,
            bias=self.bias,
            indices=self.indices,
            centroids=self.centroids,
            residual_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            num_codebooks=self.num_codebooks,
            num_centroids=self.num_centroids,
            num_residual_centroids=self.num_res_centroids,
            out_features=self.out_features,
        )

        print(self.x)
        print(out)
        assert torch.allclose(out, self.x, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
