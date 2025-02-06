# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import torch

import vptq


class TestQuantGemv(unittest.TestCase):

    def __init_serial(self, numel, device, dtype, debug=False) -> torch.Tensor:
        if (numel % 2048):
            raise ValueError("numel must be a multiple of 2048")

        n_repeats = numel // 2048
        values = list(range(2048)) * n_repeats

        x = torch.tensor(values, device=device, dtype=dtype)

        if debug:
            x_values = x.tolist()
            res = ""
            for i in range(numel):
                res += "%.0f," % x_values[i]
                if (i + 1) % 16 == 0:
                    res += "\n"
            print(res)

        return x

    def setUp(self):
        dtype = torch.bfloat16
        # dtype = torch.float16

        device = torch.device("cuda", 0)

        batch_size = 3
        length = 50

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

        # for unittest
        # numel = shape[0] * shape[1] * shape[2]
        # self.centroids = self.__init_serial(numel, device, dtype)
        # print("centroids: ", self.centroids[numel - 256:])

        self.centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (self.num_codebooks, self.num_res_centroids, self.vector_length)
        self.res_centroids = torch.randn(*shape, device=device, dtype=dtype)

        shape = (1, self.out_features)
        self.bias = torch.randn(*shape, device=device, dtype=dtype)

        # the scaling and bias tensors. NOTE, the scale weights and bias are
        # applied to the in_features in this test
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
            in_features=self.in_features,
            out_features=self.out_features,
        )
        print(self.centroids[0, 127, :])
        print(out[0, 127, :])

        assert torch.allclose(out, self.centroids, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
