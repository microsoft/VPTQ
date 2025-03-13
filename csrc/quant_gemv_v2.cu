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
    const torch::Tensor& act,
    const c10::optional<torch::Tensor>& bias,  //
    const torch::Tensor& indices,              //
    const torch::Tensor& centroids,            //
    const c10::optional<torch::Tensor>& residual_indices,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& scale_weights,
    const c10::optional<torch::Tensor>& scale_bias,  //
    int64_t out_features) {
  CHECK_INPUT(act);
  CHECK_INPUT(indices);
  CHECK_INPUT(centroids);

  const at::ScalarType dtype = act.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "the activations must be either half-precision (fp16) or bfloat16.");
  TORCH_CHECK(
      centroids.scalar_type() == dtype,
      "the main centroids must be either half-precision (fp16) or bfloat16.");

  TORCH_CHECK_EQ(act.ndimension(), 3);
  TORCH_CHECK_EQ(centroids.ndimension(), 3);

  const int64_t batch = act.size(0);
  const int64_t seq_length = act.size(1);
  const int64_t in_features = act.size(2);

  const int64_t num_codebooks = centroids.size(0);
  const int64_t num_centroids = centroids.size(1);
  const int64_t vec_len = centroids.size(2);

  TORCH_CHECK_LT(batch * seq_length, 16)
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
    // once `residual_centroids` has value, `residual_indices`
    // must have value as well
    CHECK_INPUT(residual_indices.value());
  }

  if (scale_weights.has_value()) {
    CHECK_INPUT(scale_weights.value());
    CHECK_INPUT(scale_bias.value());

    TORCH_CHECK(
        scale_weights.value().scalar_type() == dtype,
        "the scale weights must be either half-precision (fp16) or bfloat16.");
    TORCH_CHECK(
        scale_bias.value().scalar_type() == dtype,
        "the scale bias must be either half-precision (fp16) or bfloat16.");
  }

  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias.value().scalar_type() == dtype,
                "the bias must be either half-precision (fp16) or bfloat16.");
  }

  torch::Tensor output;
  output = at::empty({batch, seq_length, out_features}, centroids.options());

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int block_z = divup<int64_t, int64_t, int64_t>(out_features, vec_len);
  dim3 blocks(batch * seq_length, num_codebooks, block_z);

  // FIXME(ying): refine the choice of threads in a thread
  // block. For test at the moment.
  static const int kThreads = 4 * WARP_SIZE;
  dim3 threads(kThreads, 1, 1);

  // TODO(ying): this is hardware dependent. Need to make it
  // adaptive.
  const int kMaxSmemPerBlock = 48 * 1024;

  VPTQ_DISPATCH_TYPES(dtype, [&] {
    VPTQ_DISPATCH_VEC_LENGTH(vec_len, [&] {
      VPTQ_DISPATCH_NUM_CENTROIDS(num_centroids, [&] {
        VPTQ_DISPATCH_RES_NUM_CENTROIDS(num_res_centroids, [&] {
          const DType* residual_centroids_ptr =
              residual_centroids.has_value()
                  ? reinterpret_cast<const DType*>(
                        residual_centroids.value().data_ptr())
                  : nullptr;

          const ResIdType* residual_indices_ptr =
              residual_indices.has_value()
                  ? reinterpret_cast<const ResIdType*>(
                        residual_indices.value().data_ptr())
                  : nullptr;

          const DType* bias_ptr =
              bias.has_value()
                  ? reinterpret_cast<const DType*>(bias.value().data_ptr())
                  : nullptr;

          const DType* scale_weights_ptr =
              scale_weights.has_value() ? reinterpret_cast<const DType*>(
                                              scale_weights.value().data_ptr())
                                        : nullptr;

          const DType* scale_bias_ptr = scale_bias.has_value()
                                            ? reinterpret_cast<const DType*>(
                                                  scale_bias.value().data_ptr())
                                            : nullptr;

          static constexpr int kTileSize = 512;

          // NOTE: IdType and ResIdType are declared in the
          // dispatch macros according to
          using Config =
              kernels::QuantGemvKeTraits<DType, IdType, ResIdType, kThreads,
                                         kTileSize, kVecLen, kNumCentroids,
                                         kNumResCentroids>;
          using SharedStorage = Config::SharedStorage;
          int smem_size = SharedStorage::kSmemSize;

          std::cout << "kIdsPerBank: " << Config::kIdsPerBank << std::endl;

          auto kernel =
              &kernels::ke_quant_gemv_v2<DType, IdType, ResIdType,
                                         Config::SharedStorage, Config>;

          // TODO(ying): Check whether shared memory usage
          // exceeds the hardware limit.
          if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
          }

          kernel<<<blocks, threads, smem_size, stream>>>(
              reinterpret_cast<DType*>(output.mutable_data_ptr()),
              reinterpret_cast<const DType*>(act.data_ptr()), bias_ptr,
              indices.data_ptr<IdType>(),
              reinterpret_cast<const DType*>(centroids.data_ptr()),
              residual_indices_ptr, residual_centroids_ptr, scale_weights_ptr,
              scale_bias_ptr, batch, seq_length, in_features, out_features);
        });
      });
    });
  });

  return output;
}
}  // namespace vptq
