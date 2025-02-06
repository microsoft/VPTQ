// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quant_gemv_v2.cuh"
#include "util/common.h"
#include "util/cuda_utils.cuh"
#include "util/dispatch_macros.h"

namespace vptq {

/**
 * @brief Quantized GEMV kernel.
 * @param act The input activations.
 * @param bias The bias.
 * @param indices The indices.
 * @param centroids The codebook for the main vector quantized weights.
 *        Stored in row-major order. Element type: fp16, bf16.
 *        Shape: (num_codebooks, num_centroids, vec_len).
 * @param residual_centroids The residual centroids.
 * @param scale_weights The scale weights.
 * @param scale_bias The scale bias.
 * @param in_features The number of input features.
 * @param out_features The number of output features.
 */
torch::Tensor quant_gemv_v2(
    const torch::Tensor& act, const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& indices, const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& residual_centroids,
    const torch::Tensor& scale_weights, const torch::Tensor& scale_bias,
    int64_t in_features, int64_t out_features) {
  CHECK_INPUT(act);
  CHECK_INPUT(indices);
  CHECK_INPUT(centroids);
  CHECK_INPUT(scale_weights);
  CHECK_INPUT(scale_bias);

  const at::ScalarType dtype = act.scalar_type();

  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "the activations require the data type to be either "
      "half-precision (fp16) or bfloat16.");

  TORCH_CHECK(centroids.scalar_type() == dtype &&
                  scale_weights.scalar_type() == dtype &&
                  scale_bias.scalar_type() == dtype,
              "the centroids, scale weights and scale bias require the data "
              "type to be either half-precision (fp16) or bfloat16.");

  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias.value().scalar_type() == dtype,
                "the bias requires the data type to be either "
                "half-precision (fp16) or bfloat16.");
  }

  TORCH_CHECK_EQ(act.ndimension(), 3);
  TORCH_CHECK_EQ(centroids.ndimension(), 3);

  const int64_t batch = act.size(0);
  const int64_t num_codebooks = centroids.size(0);
  const int64_t num_centroids = centroids.size(1);
  const int64_t vec_len = centroids.size(2);

  TORCH_CHECK_LT(batch, 16)
      << "In GEMV, the batch size is suggested to be less than 16.";

  TORCH_CHECK_EQ(num_codebooks, 1) << "Only support one codebook.";

  TORCH_CHECK(
      vec_len == 4 || vec_len == 8 || vec_len == 12 || vec_len == 16,
      "Supported vector length in vectorized quantization: 4, 8, 12, or 16.");

  torch::Tensor output;
  output = at::empty({in_features, out_features}, centroids.options());

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int block_z = cuda::ceil_div(out_features, vec_len);

  dim3 blocks(batch, num_codebooks, block_z);
  // FIXME(ying): refine the choice of threads in a thread block.
  // For test at the moment.
  dim3 threads(256, 1, 1);  // four warps in a thread block.

  std::cout << "num_codebooks: " << num_codebooks << std::endl
            << "num_centroids: " << num_centroids << std::endl
            << "vec_len: " << vec_len << std::endl;

  VPTQ_DISPATCH_TYPES(dtype, [&] {
    const nv_type* residual_centroids_ptr =
        residual_centroids.has_value()
            ? reinterpret_cast<const nv_type*>(
                  residual_centroids.value().data_ptr())
            : nullptr;

    const nv_type* bias_ptr =
        bias.has_value() ? reinterpret_cast<nv_type*>(bias.value().data_ptr())
                         : nullptr;

    quant_gemv_v2_kernel<nv_type><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<nv_type*>(output.mutable_data_ptr()),
        reinterpret_cast<const nv_type*>(act.data_ptr()), bias_ptr,
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const nv_type*>(centroids.data_ptr()),
        residual_centroids_ptr,
        reinterpret_cast<const nv_type*>(scale_weights.data_ptr()),
        reinterpret_cast<const nv_type*>(scale_bias.data_ptr()), in_features,
        out_features, vec_len);
  });

  return output;
}
}  // namespace vptq
