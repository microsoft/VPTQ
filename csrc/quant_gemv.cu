// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include "quant_gemv.cuh"

namespace vptq {

#define CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce,      \
                        ResidualBits)                                          \
  {                                                                            \
    using nv_type = typename C10ToNvType<scalar_t>::type;                      \
    WqA16WithOutliers_PackIndice<nv_type, IDXBITS, ResidualBits, BASEGROUP, 4, \
                                 Do_Reduce>                                    \
        <<<blocks, threads, shared_memory_size, stream>>>(                     \
            reinterpret_cast<nv_type*>(out_buf.data_ptr<scalar_t>()),          \
            reinterpret_cast<const nv_type*>(input.data_ptr<scalar_t>()),      \
            q_indice.data_ptr<int32_t>(), outliers_indices_ptr,                \
            reinterpret_cast<const nv_type*>(centroids.data_ptr<scalar_t>()),  \
            residual_centroids.has_value()                                     \
                ? reinterpret_cast<const nv_type*>(                            \
                      residual_centroids.value().data_ptr<scalar_t>())         \
                : nullptr,                                                     \
            outliers_centroids.has_value()                                     \
                ? reinterpret_cast<const nv_type*>(                            \
                      outliers_centroids.value().data_ptr<scalar_t>())         \
                : nullptr,                                                     \
            perm_ptr,                                                          \
            reinterpret_cast<const nv_type*>(                                  \
                weight_scale.data_ptr<scalar_t>()),                            \
            reinterpret_cast<const nv_type*>(                                  \
                weight_bias.data_ptr<scalar_t>()),                             \
            bias.has_value() ? reinterpret_cast<const nv_type*>(               \
                                   bias.value().data_ptr<scalar_t>())          \
                             : nullptr,                                        \
            out_features, in_features, outliers_indices_size_n1,               \
            q_indice.stride(0), q_indice.stride(1), centroids.stride(0),       \
            q_indice.size(0));                                                 \
  }

#define CallWqA16kernel_dtype(out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                              ResidualBits)                           \
  if (input.dtype() == at::ScalarType::Half) {                        \
    using scalar_t = c10::Half;                                       \
    CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                    ResidualBits);                                    \
  } else {                                                            \
    using scalar_t = c10::BFloat16;                                   \
    CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                    ResidualBits);                                    \
  }

#define CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, ResidualBits)     \
  switch (index_bits) {                                                       \
    case 16:                                                                  \
      CallWqA16kernel_dtype(out_buf, 16, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 15:                                                                  \
      CallWqA16kernel_dtype(out_buf, 15, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 14:                                                                  \
      CallWqA16kernel_dtype(out_buf, 14, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 13:                                                                  \
      CallWqA16kernel_dtype(out_buf, 13, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 12:                                                                  \
      CallWqA16kernel_dtype(out_buf, 12, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 10:                                                                  \
      CallWqA16kernel_dtype(out_buf, 10, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 8:                                                                   \
      CallWqA16kernel_dtype(out_buf, 8, BASEGROUP, Do_Reduce, ResidualBits);  \
      break;                                                                  \
    case 4:                                                                   \
      CallWqA16kernel_dtype(out_buf, 4, BASEGROUP, Do_Reduce, ResidualBits);  \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false,                                                      \
                  "un-supported index_bits:" + std::to_string(index_bits));   \
  }

#define DispatchWqA16Kernel(out_buf, BASEGROUP, Do_Reduce)     \
  switch (res_index_bits) {                                    \
    case 16:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 16); \
      break;                                                   \
    case 15:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 15); \
      break;                                                   \
    case 12:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 12); \
      break;                                                   \
    case 11:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 11); \
      break;                                                   \
    case 10:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 10); \
      break;                                                   \
    case 9:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 9);  \
      break;                                                   \
    case 8:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 8);  \
      break;                                                   \
    case 7:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 7);  \
      break;                                                   \
    case 6:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 6);  \
      break;                                                   \
    case 5:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 5);  \
      break;                                                   \
    case 4:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 4);  \
      break;                                                   \
    case 3:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 3);  \
      break;                                                   \
    case 2:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 2);  \
      break;                                                   \
    case 0:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 0);  \
      break;                                                   \
    default:                                                   \
      TORCH_CHECK(false, "un-supported res_index_bits:" +      \
                             std::to_string(res_index_bits));  \
  }

// @brief launch_gemv_outliers_cuda_packkernel
// @param out_features
// @param input
// @param q_indice [num_cen, o_c_size, in_inf]
// @param centroids, Tensor[fp16|bf16], a 3-D tensor with a shape of
//        (num_codebooks, num_centroids, vector_len).
// @param q_indice_residual, Tensor[fp16|bf16], a 3-D tensor with a shape of
//        (num_codebooks, num_residual_centroids, vector_len)
// @param residual_centroids [num_c, c_size, vec_len]
// @param outliers_indices [num_cen, c_size, ol_in_f]
// @param outliers_centroids [num_c, c_size, out_vec_len]
// @param perm
// @param weight_scale
// @param weight_bias
// @param bias
// @return torch::Tensor
torch::Tensor launch_gemv_outliers_cuda_packkernel(
    const int64_t out_features,  //
    const torch::Tensor& input, const torch::Tensor& q_indice,
    const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& outliers_indices,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias,
    const c10::optional<torch::Tensor>& bias) {
  OptionalCUDAGuard cudaguard(input.device().index());

  // the length of a vector in vector quantization
  const int64_t base_groupsize = centroids.size(-1);

  int index_bits = log2(centroids.size(1));
  int res_index_bits = residual_centroids.has_value()
                           ? log2(residual_centroids.value().size(1))
                           : 0;

  const int in_features = input.size(-1);

  auto output_shape = input.sizes().vec();
  output_shape[input.dim() - 1] = out_features;
  torch::Tensor output;

  dim3 blocks(cuda::ceil_div(out_features, base_groupsize),
              input.numel() / in_features);
  dim3 threads(cuda::kBlockSize);  // 256 threads = 8 warps
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int shared_memory_size = 2 * in_features * 2;
  const int outliers_indices_size_n1 =
      outliers_indices.has_value() ? outliers_indices.value().size(-1) : 0;

  if (outliers_centroids.has_value()) {
    TORCH_CHECK(outliers_centroids.value().size(-1) == 4,
                "only support 4 out_vec_len");
  }
  const uint16_t* outliers_indices_ptr =
      (const uint16_t*)(outliers_indices.has_value()
                            ? outliers_indices.value().data_ptr<int16_t>()
                            : nullptr);
  const uint16_t* perm_ptr =
      perm.has_value() ? (const uint16_t*)(perm.value().data_ptr<int16_t>())
                       : nullptr;

  if (in_features <= cuda::kBlockSize) {
    TORCH_CHECK(false, "un-supported yet");
  } else {
    constexpr int do_reduce = 4;
    shared_memory_size = 0;
    auto tmp_output_shape = output_shape;
    tmp_output_shape.push_back(
        cuda::ceil_div(in_features, cuda::kBlockSize * do_reduce));
    torch::Tensor tmp_output = at::empty(tmp_output_shape, centroids.options());
    blocks.z = tmp_output_shape.back();

    switch (base_groupsize) {
      case 16:
        DispatchWqA16Kernel(tmp_output, 16, do_reduce);
        break;
      case 12:
        DispatchWqA16Kernel(tmp_output, 12, do_reduce);
        break;
      case 10:
        DispatchWqA16Kernel(tmp_output, 10, do_reduce);
        break;
      case 8:
        DispatchWqA16Kernel(tmp_output, 8, do_reduce);
        break;
      case 6:
        DispatchWqA16Kernel(tmp_output, 6, do_reduce);
        break;
      case 4:
        DispatchWqA16Kernel(tmp_output, 4, do_reduce);
        break;
      case 2:
        DispatchWqA16Kernel(tmp_output, 2, do_reduce);
        break;
      default:
        TORCH_CHECK(false,
                    "un-supported groupsize:" + std::to_string(base_groupsize));
    }
    output = tmp_output.sum(-1);
  }

  return output;
}

torch::Tensor wquant_act16_gemv(
    const torch::Tensor& input,  // activation
    const torch::Tensor& q_indice, const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& q_indice_outliers,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& invperm,
    const torch::Tensor& weight_scale, const torch::Tensor& weight_bias,
    const c10::optional<torch::Tensor>& bias,  //
    int64_t in_features, int64_t out_features) {
  CHECK_INPUT(q_indice);
  CHECK_INPUT(input);
  if (q_indice_residual.has_value()) {
    CHECK_INPUT(q_indice_residual.value());
    CHECK_INPUT(residual_centroids.value());
  }
  TORCH_CHECK_EQ(q_indice.dtype(), torch::kInt)
      << "`q_indice` must be integers.";

  CHECK_INPUT(centroids);
  auto dev_index = q_indice.device().index();
  auto inv_perm_device_index = dev_index;

  if (invperm.has_value()) {
    CHECK_INPUT(invperm.value());
    inv_perm_device_index = invperm.value().device().index();
  }

  if (q_indice_residual.has_value()) {
    TORCH_CHECK_EQ(centroids.sizes(), residual_centroids.value().sizes())
        << "The numel of centroids and residual_centroids must be the same.";

    TORCH_CHECK_EQ(q_indice_residual.value().device().index(), dev_index)
        << "the residual index tensor is on a different device.";
    TORCH_CHECK_EQ(centroids.device().index(), dev_index)
        << "the centroids tensor is on a different device.";
    TORCH_CHECK_EQ(residual_centroids.value().device().index(), dev_index)
        << "the residual centroids tensor is on a different device.";
    TORCH_CHECK_EQ(inv_perm_device_index, dev_index)
        << "the inverted permutation index tensor is on a different device.";
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;

  output = launch_gemv_outliers_cuda_packkernel(
      out_features, input, q_indice, centroids, q_indice_residual,
      residual_centroids, q_indice_outliers, outliers_centroids, invperm,
      weight_scale, weight_bias, bias);

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

}  // namespace vptq
