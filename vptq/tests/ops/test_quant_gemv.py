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
        self.res_indices = torch.as_tensor(
            list(range(self.num_res_centroids)) * num_repeats,
            device=device,
            dtype=torch.uint8
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

    def compare_float_tensors(self, tensor1, tensor2):
        # For bfloat16 tensors, we need to convert to float32 for accurate
        # comparison since bfloat16 has limited precision.

        if tensor1.dtype == torch.bfloat16:
            tensor1_float = tensor1.float()
            tensor2_float = tensor2.float()

            self.assertEqual(
                tensor1_float.shape, tensor2_float.shape,
                "Tensor shapes don't match"
            )

            rtol, atol = 1e-2, 1e-2
            self.assertTrue(
                torch.allclose(
                    tensor1_float, tensor2_float, rtol=rtol, atol=atol
                ),
                f"Tensors not equal within tolerance: rtol={rtol}, atol={atol}"
            )
        else:
            self.assertEqual(
                tensor1.shape, tensor2.shape, "Tensor shapes don't match"
            )
            self.assertTrue(
                torch.allclose(tensor1, tensor2), "Tensors not equal"
            )

    def compare_int_tensors(self, tensor1, tensor2):
        self.assertEqual(
            tensor1.shape, tensor2.shape, "Integer tensor shapes don't match"
        )

        # If the above assertion fails and we need more details:
        if not torch.all(tensor1 == tensor2):
            # Find indices where tensors differ
            diff_indices = torch.nonzero(tensor1 != tensor2)
            if len(diff_indices) > 0:
                # Get a sample of differences (up to 5)
                sample_size = min(5, len(diff_indices))
                for i in range(sample_size):
                    idx = tuple(diff_indices[i].tolist())
                    print((
                        "Difference at index "
                        f"{idx}: {tensor1[idx]} vs {tensor2[idx]}"
                    ))

        self.assertEqual(
            tensor1.dtype, tensor2.dtype, (
                "Integer tensor dtypes don't match: "
                f"{tensor1.dtype} vs {tensor2.dtype}"
            )
        )

        # For integer tensors, we want exact equality
        self.assertTrue(
            torch.all(tensor1 == tensor2),
            "Integer tensors are not exactly equal"
        )

    def test(self):
        out1 = vptq.ops.quant_gemv_v2(
            self.x,
            bias=self.bias,
            indices=self.main_indices,
            centroids=self.centroids,
            residual_indices=self.res_indices,
            residual_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            num_codebooks=self.num_codebooks,
            num_centroids=self.num_centroids,
            num_residual_centroids=self.num_res_centroids,
            out_features=self.out_features
        )
        print(out1[0, 0, 0:32])

        out2 = ground_truth(
            x=self.x,
            bias=self.bias,
            indices=self.main_indices,
            centroids=self.centroids,
            residual_indices=self.res_indices,
            residual_centroids=self.res_centroids,
            scale_weights=self.scale_weights,
            scale_bias=self.scale_bias,
            vector_len=self.vector_length,
            out_features=self.out_features
        )
        print(out2[0, 0, 0:32])

        self.compare_float_tensors(out1, out2)


if __name__ == "__main__":
    unittest.main()
