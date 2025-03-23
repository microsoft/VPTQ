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
    res_indices: torch.Tensor,
    res_centroids: torch.Tensor,
    scale_weights: torch.Tensor,
    scale_bias: torch.Tensor,
    vector_len: int,
    out_features: int,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype

    batch_size, length, in_features = x.shape
    num_decoding = indices.shape[0]

    if num_decoding != res_indices.shape[0]:
        raise ValueError(
            ("indices and residual_indices must have the same shape.")
        )
    if num_decoding != in_features * out_features // vector_len:
        raise ValueError((
            "indices must have the same shape as "
            "in_features * out_features // vector_len."
        ))

    # construct dequantized weights
    shape = (in_features, out_features)
    main_weights = torch.zeros(shape, dtype=dtype, device=device)
    res_weights = torch.zeros(shape, dtype=dtype, device=device)

    res_indices = res_indices.to(torch.uint16)

    for i in range(num_decoding):
        row = i % in_features
        col = i // in_features * vector_len

        ids = indices[i]
        main_weights[row, col:col + vector_len] = centroids[0, ids, :]

        res_ids = res_indices[i]
        res_weights[row, col:col + vector_len] = res_centroids[0, res_ids, :]

    weights = main_weights + res_weights
    weights = scale_weights * weights + scale_bias

    out = torch.zeros(
        batch_size, length, out_features, dtype=dtype, device=device
    )
    for i in range(batch_size):
        for j in range(length):
            vec = x[i, j, :]
            out[i, j, :] = vec @ weights

    if bias is not None:
        out += bias
    return out


class TestQuantGemv(unittest.TestCase):
    mean = 2e-2
    std = 0.5

    dtype = torch.bfloat16
    device = torch.device("cuda", 0)

    def setUp(self):
        torch.manual_seed(1234)

        # gemv requires batch size * length < 16
        batch_size = 1
        length = 1

        self.in_features = 1024
        self.out_features = 2048

        self.num_codebooks = 1
        self.vector_length = 8

        self.num_centroids = 8192  # must be stored in uint16
        self.num_res_centroids = 256  # can be stored in uint8

        def _create_indices(num_indices, num_centroids, target_dtype):
            indices = torch.arange(
                num_centroids,
                device=TestQuantGemv.device,
                dtype=torch.int32,
            )
            return indices.repeat(num_indices // num_centroids
                                 ).to(dtype=target_dtype)

        def _create_tensor(size: tuple[int, ...]) -> torch.Tensor:
            return torch.normal(
                mean=TestQuantGemv.mean,
                std=TestQuantGemv.std,
                size=size,
                device=TestQuantGemv.device,
                dtype=TestQuantGemv.dtype,
            )

        # ====== generate data for unittest.  ======#
        self.x = _create_tensor((batch_size, length, self.in_features))

        # Create indices tensors
        num_indices = self.in_features * self.out_features // self.vector_length

        self.main_indices = _create_indices(
            num_indices, self.num_centroids, torch.uint16
        )
        self.res_indices = _create_indices(
            num_indices, self.num_res_centroids, torch.uint8
        )

        self.centroids = _create_tensor(
            (self.num_codebooks, self.num_centroids, self.vector_length)
        )

        self.res_centroids = _create_tensor(
            (self.num_codebooks, self.num_res_centroids, self.vector_length)
        )

        # self.bias = _create_tensor((1, 1, self.out_features))
        self.bias = None

        # NOTE: In this test, the scale weights and bias are applied
        # to the in_features.
        self.scale_weights = _create_tensor((self.in_features, 1))
        self.scale_bias = _create_tensor((self.in_features, 1))

    def compare_float_tensors(self, tensor1, tensor2):
        # For bfloat16 tensors, we need to convert to float32 for accurate
        # comparison since bfloat16 has limited precision.

        if tensor1.dtype == torch.bfloat16 or tensor1.dtype == torch.float16:
            tensor1_float = tensor1.float()
            tensor2_float = tensor2.float()

            self.assertEqual(
                tensor1_float.shape,
                tensor2_float.shape,
                "Tensor shapes don't match",
            )

            rtol, atol = 0.2, 0.2
            self.assertTrue(
                torch.allclose(
                    tensor1_float, tensor2_float, rtol=rtol, atol=atol
                ),
                f"Tensors not equal within tolerance: rtol={rtol}, atol={atol}",
            )
        else:
            self.assertEqual(
                tensor1.shape, tensor2.shape, "Tensor shapes don't match"
            )
            self.assertTrue(
                torch.allclose(tensor1, tensor2), "Tensors not equal"
            )

    def test(self):
        gemv_args = {
            "x": self.x,
            "bias": self.bias,
            "indices": self.main_indices,
            "centroids": self.centroids,
            "residual_indices": self.res_indices,
            "residual_centroids": self.res_centroids,
            "scale_weights": self.scale_weights,
            "scale_bias": self.scale_bias,
            "vector_len": self.vector_length,
            "num_codebooks": self.num_codebooks,
            "num_centroids": self.num_centroids,
            "num_residual_centroids": self.num_res_centroids,
            "out_features": self.out_features,
        }
        out1 = vptq.ops.quant_gemv_v2(**gemv_args)

        out2 = ground_truth(
            x=self.x,
            bias=self.bias,
            indices=self.main_indices,
            centroids=self.centroids,
            res_indices=self.res_indices,
            res_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            out_features=self.out_features,
        )

        self.compare_float_tensors(out1, out2)

        start = 128
        end = start + 16
        print("gemv kernel:")
        print(out1[0, 0, start:end])

        print("ground truth:")
        print(out2[0, 0, start:end])


if __name__ == "__main__":
    unittest.main()
