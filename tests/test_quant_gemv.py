# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for quantized GEMV (General Matrix-Vector multiplication) operations.

To run these tests, execute the following command from the project root:
    pytest tests/test_quant_gemv.py -v -s

The -v flag enables verbose output and -s allows print statements to be
displayed.
"""

import pytest
import torch

import vptq


def _create_indices(
    num_indices, num_centroids, device: torch.device
) -> torch.Tensor:
    target_dtype = torch.uint16 if num_centroids > 256 else torch.uint8
    indices = torch.arange(
        num_centroids,
        device=device,
        dtype=torch.int32,
    )
    return indices.repeat(num_indices // num_centroids).to(dtype=target_dtype)


def _create_tensor(
    size: tuple[int, ...],
    mean: float,
    std: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.normal(
        mean=mean,
        std=std,
        size=size,
        device=device,
        dtype=dtype,
    )


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
        raise ValueError(
            (
                "indices must have the same shape as "
                "in_features * out_features // vector_len."
            )
        )

    # construct dequantized weights
    shape = (in_features, out_features)
    main_weights = torch.zeros(shape, dtype=dtype, device=device)
    res_weights = torch.zeros(shape, dtype=dtype, device=device)

    res_indices = res_indices.to(torch.uint16)

    for i in range(num_decoding):
        row = i % in_features
        col = i // in_features * vector_len

        ids = indices[i]
        main_weights[row, col : col + vector_len] = centroids[0, ids, :]

        res_ids = res_indices[i]
        res_weights[row, col : col + vector_len] = res_centroids[0, res_ids, :]

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


@pytest.fixture
def test_data(request):
    """Fixture to generate test data with configurable parameters."""
    torch.manual_seed(1234)

    # Get parameters from the test request
    params = request.param

    # Test parameters
    batch_size = params.get("batch_size", 1)
    length = params.get("length", 1)
    in_features = params.get("in_features", 1024)
    out_features = params.get("out_features", 2048)
    num_codebooks = params.get("num_codebooks", 1)
    vector_length = params.get("vector_length", 8)
    num_centroids = params.get("num_centroids", 8192)
    num_res_centroids = params.get("num_res_centroids", 256)
    mean = params.get("mean", 2e-2)
    std = params.get("std", 0.5)
    dtype = params.get("dtype", torch.bfloat16)
    device = params.get("device", torch.device("cuda", 0))

    # Generate test data
    x = _create_tensor(
        (batch_size, length, in_features), mean, std, dtype, device
    )

    centroids = _create_tensor(
        (num_codebooks, num_centroids, vector_length), mean, std, dtype, device
    )
    res_centroids = _create_tensor(
        (num_codebooks, num_res_centroids, vector_length),
        mean,
        std,
        dtype,
        device,
    )
    bias = None
    scale_weights = _create_tensor((in_features, 1), mean, std, dtype, device)
    scale_bias = _create_tensor((in_features, 1), mean, std, dtype, device)

    num_indices = in_features * out_features // vector_length
    main_indices = _create_indices(num_indices, num_centroids, device)
    res_indices = _create_indices(num_indices, num_res_centroids, device)

    return {
        "x": x,
        "bias": bias,
        "indices": main_indices,
        "centroids": centroids,
        "residual_indices": res_indices,
        "residual_centroids": res_centroids,
        "scale_weights": scale_weights,
        "scale_bias": scale_bias,
        "vector_len": vector_length,
        "num_codebooks": num_codebooks,
        "num_centroids": num_centroids,
        "num_residual_centroids": num_res_centroids,
        "out_features": out_features,
    }


def compare_float_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """Helper function to compare tensors with appropriate precision."""
    if tensor1.dtype == torch.bfloat16 or tensor1.dtype == torch.float16:
        tensor1_float = tensor1.float()
        tensor2_float = tensor2.float()

        if tensor1_float.shape != tensor2_float.shape:
            raise ValueError("Tensor shapes don't match")

        rtol, atol = 0.2, 0.2
        if not torch.allclose(
            tensor1_float, tensor2_float, rtol=rtol, atol=atol
        ):
            raise ValueError(
                f"Tensors not equal within tolerance: rtol={rtol}, atol={atol}"
            )
    else:
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor shapes don't match")
        if not torch.allclose(tensor1, tensor2):
            raise ValueError("Tensors not equal")


test_configs = [
    # residual indices are stored in uint8
    {
        "name": "residual_indices_uint8",
        "batch_size": 1,
        "length": 1,
        "in_features": 1024,
        "out_features": 2048,
        "vector_length": 8,
        "num_centroids": 8192,
        "num_res_centroids": 256,  #  indices are stored in uint8
    },
    # residual indices are stored in uint16
    {
        "name": "residual_indices_uint16",
        "batch_size": 1,
        "length": 1,
        "in_features": 1024,
        "out_features": 1024,
        "vector_length": 8,
        "num_centroids": 8192,
        "num_res_centroids": 512,  #  indices are stored in uint16
    },
]


@pytest.mark.parametrize("test_data", test_configs, indirect=True)
def test_quant_gemv(test_data):
    """
    Test the quant_gemv operation against ground truth with different
    configurations.
    """

    # Run ground truth
    out1 = ground_truth(
        x=test_data["x"],
        bias=test_data["bias"],
        indices=test_data["indices"],
        centroids=test_data["centroids"],
        res_indices=test_data["residual_indices"],
        res_centroids=test_data["residual_centroids"],
        scale_weights=test_data["scale_weights"],
        scale_bias=test_data["scale_bias"],
        vector_len=test_data["vector_len"],
        out_features=test_data["out_features"],
    )

    # Run the quant_gemv operation
    out2 = vptq.ops.quant_gemv_v2(**test_data)

    # Compare results
    compare_float_tensors(out1, out2)

    # Print sample outputs for debugging
    start, end = 128, 144
    print(f"\nTest configuration: {test_data.get('name', 'unnamed')}")
    print("gemv kernel:")
    print(out1[0, 0, start:end])
    print("ground truth:")
    print(out2[0, 0, start:end])


def run_test_with_config(config):
    """Run a single test with the given configuration."""
    print(f"\nRunning test with configuration: {config['name']}")

    # Generate test data
    test_data = create_test_data(config)

    # Run the quant_gemv operation
    out1 = vptq.ops.quant_gemv_v2(**test_data)

    # Run ground truth
    out2 = ground_truth(
        x=test_data["x"],
        bias=test_data["bias"],
        indices=test_data["indices"],
        centroids=test_data["centroids"],
        res_indices=test_data["residual_indices"],
        res_centroids=test_data["residual_centroids"],
        scale_weights=test_data["scale_weights"],
        scale_bias=test_data["scale_bias"],
        vector_len=test_data["vector_len"],
        out_features=test_data["out_features"],
    )

    # Compare results
    try:
        compare_float_tensors(out1, out2)
        print("Test passed!")
    except ValueError as e:
        print(f"Test failed: {str(e)}")

    # Print sample outputs for debugging
    start, end = 128, 144
    print("\nSample outputs:")
    print("gemv kernel:")
    print(out1[0, 0, start:end])
    print("ground truth:")
    print(out2[0, 0, start:end])
    print("-" * 80)


def create_test_data(config):
    """Factory function to generate test data from config."""
    torch.manual_seed(1234)

    # Test parameters
    batch_size = config.get("batch_size", 1)
    length = config.get("length", 1)
    in_features = config.get("in_features", 1024)
    out_features = config.get("out_features", 2048)
    num_codebooks = config.get("num_codebooks", 1)
    vector_length = config.get("vector_length", 8)
    num_centroids = config.get("num_centroids", 8192)
    num_res_centroids = config.get("num_res_centroids", 256)
    mean = config.get("mean", 2e-2)
    std = config.get("std", 0.5)
    dtype = config.get("dtype", torch.bfloat16)
    device = config.get("device", torch.device("cuda", 0))

    # Generate test data
    x = _create_tensor(
        (batch_size, length, in_features), mean, std, dtype, device
    )
    centroids = _create_tensor(
        (num_codebooks, num_centroids, vector_length), mean, std, dtype, device
    )
    res_centroids = _create_tensor(
        (num_codebooks, num_res_centroids, vector_length),
        mean,
        std,
        dtype,
        device,
    )
    bias = None
    scale_weights = _create_tensor((in_features, 1), mean, std, dtype, device)
    scale_bias = _create_tensor((in_features, 1), mean, std, dtype, device)

    num_indices = in_features * out_features // vector_length
    main_indices = _create_indices(num_indices, num_centroids, device)
    res_indices = _create_indices(num_indices, num_res_centroids, device)

    return {
        "x": x,
        "bias": bias,
        "indices": main_indices,
        "centroids": centroids,
        "residual_indices": res_indices,
        "residual_centroids": res_centroids,
        "scale_weights": scale_weights,
        "scale_bias": scale_bias,
        "vector_len": vector_length,
        "num_codebooks": num_codebooks,
        "num_centroids": num_centroids,
        "num_residual_centroids": num_res_centroids,
        "out_features": out_features,
    }


def main():
    """Run tests directly without pytest."""
    print("Running VPTQ quant_gemv tests...")
    print("=" * 80)

    # Run all configurations
    for config in test_configs:
        run_test_with_config(config)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
