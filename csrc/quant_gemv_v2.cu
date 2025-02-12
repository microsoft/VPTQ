// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dispatch_macros.h"
#include "kernels/quant_gemv_traits.cuh"
#include "kernels/quant_gemv_v2.cuh"
#include "util/common.h"
#include "util/math_utils.h"
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
      vec_len == 4 || vec_len == 8 || vec_len == 16,
      "Supported vector length in vectorized quantization: 4, 8, or 16.");

  int64_t num_res_centroids = 0;
  if (residual_centroids.has_value()) {
    CHECK_INPUT(residual_centroids.value());
    TORCH_CHECK_EQ(residual_centroids.value().ndimension(), 3);
    TORCH_CHECK_EQ(residual_centroids.value().size(0), 1)
        << "Only support one codebook.";
    TORCH_CHECK_EQ(residual_centroids.value().size(2), vec_len)
        << "The vector length of the residual centroids must be the same as "
           "the main centroids.";

    num_res_centroids = residual_centroids.value().size(1);
  }

  torch::Tensor output;
  output = at::empty({in_features, out_features}, centroids.options());

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int block_z = divup<int64_t, int64_t, int64_t>(out_features, vec_len);
  dim3 blocks(batch, num_codebooks, block_z);

  // FIXME(ying): refine the choice of threads in a thread block.
  // For test at the moment.
  static const int kThreads = 2 * 4 * WARP_SIZE;
  dim3 threads(kThreads, 1, 1);  // 8 warps in a thread block.

  // TODO(ying): this is hardware dependent. Need to make it adaptive.
  const int kMaxSmemPerBlock = 48 * 1024;

  VPTQ_DISPATCH_TYPES(dtype, [&] {
    VPTQ_DISPATCH_VEC_LENGTH(vec_len, [&] {
      VPTQ_DISPATCH_NUM_CENTROIDS(num_centroids, [&] {
        VPTQ_DISPATCH_RES_NUM_CENTROIDS(num_res_centroids, [&] {
          const nv_type* residual_centroids_ptr =
              residual_centroids.has_value()
                  ? reinterpret_cast<const nv_type*>(
                        residual_centroids.value().data_ptr())
                  : nullptr;

          const nv_type* bias_ptr =
              bias.has_value()
                  ? reinterpret_cast<nv_type*>(bias.value().data_ptr())
                  : nullptr;

          // load entire codebooks into shared memory
          int64_t size_codebook =
              (kNumCentroids + kNumResCentroids) * kVecLen * sizeof(nv_type);
          int64_t smem_size = size_codebook;

          std::cout << "centroid number: " << kNumCentroids
                    << "; residual centroid number: " << kNumResCentroids
                    << "; vector length: " << kVecLen << ";" << std::endl
                    << "smem_size: " << smem_size / 1024 << "KB;" << std::endl;

          using Config =
              kernels::QuantGemvKeTraits<nv_type, kThreads, kNumCentroids,
                                         kNumResCentroids, kVecLen>;

          std::cout << "numel1: " << Config::MainCentroidTraits::kNumel
                    << ", numel2: " << Config::ResCentroidTraits::kNumel
                    << std::endl;

          static constexpr int kTileSize = 512;
          auto kernel = &kernels::ke_quant_gemv_v2<nv_type, Config, kTileSize>;
          // TODO(ying): Check whether shared memory usage exceeds the hardware
          // limit.
          if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
          }

          kernel<<<blocks, threads, smem_size, stream>>>(
              reinterpret_cast<nv_type*>(output.mutable_data_ptr()),
              reinterpret_cast<const nv_type*>(act.data_ptr()), bias_ptr,
              indices.data_ptr<int32_t>(),
              reinterpret_cast<const nv_type*>(centroids.data_ptr()),
              residual_centroids_ptr,
              reinterpret_cast<const nv_type*>(scale_weights.data_ptr()),
              reinterpret_cast<const nv_type*>(scale_bias.data_ptr()),
              in_features, out_features);
        });
      });
    });
  });

  return output;
}
}  // namespace vptq
