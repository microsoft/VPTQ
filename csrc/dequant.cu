// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/dequant.cuh"
#include "util/common.h"
#include "util/math_utils.h"

namespace vptq {

#define callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF,     \
                                ResidualBits)                                  \
  {                                                                            \
    using nv_type = typename C10ToNvType<scalar_t>::type;                      \
    kernels::DequantizeWithOutliers_PackIndice<nv_type, IDXBITS, ResidualBits, \
                                               BASEGROUP, OUT_OUF_INF>         \
        <<<blocks, threads, 0, stream>>>(                                      \
            reinterpret_cast<nv_type*>(output.data_ptr<scalar_t>()),           \
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
            out_size[0], out_size[1], outliers_indices_size_n1,                \
            outliers_centroids_size_n1, q_indice.stride(0),                    \
            q_indice.stride(1), centroids.stride(0), q_indice.size(0));        \
  }

#define callDequantWithOutliers_dtype(IDXBITS, BASEGROUP, OUT_OUF_INF, \
                                      ResidualBits)                    \
  if (centroids.dtype() == at::ScalarType::Half) {                     \
    using scalar_t = c10::Half;                                        \
    callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF, \
                            ResidualBits);                             \
  } else {                                                             \
    using scalar_t = c10::BFloat16;                                    \
    callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF, \
                            ResidualBits);                             \
  }

#define callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, ResidualBits)     \
  switch (index_bits) {                                                        \
    case 16:                                                                   \
      callDequantWithOutliers_dtype(16, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 15:                                                                   \
      callDequantWithOutliers_dtype(15, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 14:                                                                   \
      callDequantWithOutliers_dtype(14, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 13:                                                                   \
      callDequantWithOutliers_dtype(13, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 12:                                                                   \
      callDequantWithOutliers_dtype(12, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 8:                                                                    \
      callDequantWithOutliers_dtype(8, BASEGROUP, OUT_OUF_INF, ResidualBits);  \
      break;                                                                   \
    case 4:                                                                    \
      callDequantWithOutliers_dtype(4, BASEGROUP, OUT_OUF_INF, ResidualBits);  \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "un-supported index_bits:" + std::to_string(index_bits));    \
  }

#define CASE_callDequantWithOutliers_bits(rib)                 \
  case rib: {                                                  \
    callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, rib); \
    break;                                                     \
  }

#define DispatchDequantWithOutliers(BASEGROUP, OUT_OUF_INF)     \
  switch (res_index_bits) {                                     \
    case 16:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 16); \
      break;                                                    \
    case 15:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 15); \
      break;                                                    \
    case 12:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 12); \
      break;                                                    \
    case 11:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 11); \
      break;                                                    \
    case 10:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 10); \
      break;                                                    \
    case 9:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 9);  \
      break;                                                    \
    case 8:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 8);  \
      break;                                                    \
    case 7:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 7);  \
      break;                                                    \
    case 6:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 6);  \
      break;                                                    \
    case 5:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 5);  \
      break;                                                    \
    case 4:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 4);  \
      break;                                                    \
    case 3:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 3);  \
      break;                                                    \
    case 2:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 2);  \
      break;                                                    \
    case 0:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 0);  \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "un-supported res_index_bits:" +       \
                             std::to_string(res_index_bits));   \
  }

#define CASE_DispatchDequantWithOutliers(bgsize)      \
  case bgsize: {                                      \
    DispatchDequantWithOutliers(bgsize, out_ouf_inf); \
    break;                                            \
  }

// @brief launch_deqantize_outliers_cuda_packkernel
// @param outf_x_inf [out_f, in_f]
// @param q_indice [num_cen, o_c_size, in_inf]
// @param centroids [num_c, c_size, vec_len]
// @param q_indice_residual [num_cen, o_c_size, in_inf]
// @param residual_centroids [num_c, c_size, vec_len]
// @param outliers_indices [num_cen, c_size, ol_in_f]
// @param outliers_centroids [num_c, c_size, out_vec_len]
// @param perm
// @param weight_scale
// @param weight_bias
// @return torch::Tensor
torch::Tensor launch_deqantize_outliers_cuda_packkernel(
    const int64_t* outf_x_inf, const torch::Tensor& q_indice,
    const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& outliers_indices,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias) {
  OptionalCUDAGuard cudaguard(q_indice.device().index());
  int base_groupsize = centroids.size(-1);  // how many elements in a vector
  int res_groupsize =
      residual_centroids.has_value() ? residual_centroids.value().size(-1) : 0;
  TORCH_CHECK(((res_groupsize == base_groupsize) || (res_groupsize == 0)),
              "res_groupsize==base_groupsize is false, must be true");

  // how many bits to index quantization vector
  int index_bits = log2(centroids.size(1));
  int res_index_bits = residual_centroids.has_value()
                           ? log2(residual_centroids.value().size(1))
                           : 0;
  auto out_size = outf_x_inf;
  dim3 blocks(divup<int64_t, int64_t, int64_t>(
      divup<int64_t, int64_t, int64_t>(out_size[0], base_groupsize) *
          out_size[1],
      cuda::kBlockSize));
  dim3 threads(cuda::kBlockSize);
  torch::Tensor output;

  // FIXME: why =false is 10 times slow?
  constexpr bool out_ouf_inf = true;
  if (out_ouf_inf) {  // out_ouf_inf
    output = at::empty({out_size[0], out_size[1]}, centroids.options());
  } else {
    output = at::empty({out_size[1], out_size[0]}, centroids.options());
  }
  int outliers_indices_size_n1 =
      outliers_indices.has_value() ? outliers_indices.value().size(-1) : 0;
  int outliers_centroids_size_n1 =
      outliers_centroids.has_value() ? outliers_centroids.value().size(-1) : 1;

  const uint16_t* perm_ptr =
      perm.has_value() ? (const uint16_t*)(perm.value().data_ptr<int16_t>())
                       : nullptr;
  const int16_t* outliers_indices_ptr =
      outliers_indices.has_value()
          ? outliers_indices.value().data_ptr<int16_t>()
          : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  switch (base_groupsize) {
    CASE_DispatchDequantWithOutliers(16);
    CASE_DispatchDequantWithOutliers(12);
    CASE_DispatchDequantWithOutliers(8);
    CASE_DispatchDequantWithOutliers(6);
    CASE_DispatchDequantWithOutliers(4);
    CASE_DispatchDequantWithOutliers(2);
    default:
      TORCH_CHECK(false, "un-supported base_groupsize:" +
                             std::to_string(base_groupsize));
  }

#undef CASE_DispatchDequantWithOutliers

  if (out_ouf_inf) {
    return output;
  } else {
    return output.t();
  }
}

torch::Tensor dequant(const torch::Tensor& q_indice,
                      const torch::Tensor& centroids,
                      const c10::optional<torch::Tensor>& q_indice_residual,
                      const c10::optional<torch::Tensor>& residual_centroids,
                      const c10::optional<torch::Tensor>& q_indice_outliers,
                      const c10::optional<torch::Tensor>& outliers_centroids,
                      const c10::optional<torch::Tensor>& invperm,
                      const torch::Tensor& weight_scale,
                      const torch::Tensor& weight_bias, int64_t groupsize,
                      int64_t in_features, int64_t out_features) {
  auto dev_index = q_indice.device().index();

  CHECK_INPUT(q_indice);
  if (q_indice_residual.has_value()) {
    CHECK_INPUT(q_indice_residual.value());
    CHECK_INPUT(residual_centroids.value());
  }

  CHECK_INPUT(centroids);
  auto perm_dev_index = dev_index;
  if (invperm.has_value()) {
    CHECK_INPUT(invperm.value());
    perm_dev_index = invperm.value().device().index();
  }
  CHECK_INPUT(weight_scale);
  CHECK_INPUT(weight_bias);
  TORCH_CHECK_EQ(q_indice.dtype(), torch::kInt)
      << "`q_indice` must have a type of integer.";
  TORCH_CHECK_GE(groupsize, 2) << "groupsize must be >= 2.";
  TORCH_CHECK_EQ(q_indice.dim(), 3) << "`q_indice` must be a 3D tensor.";

  if (q_indice_residual.has_value()) {
    TORCH_CHECK_EQ(q_indice.size(0), centroids.size(0))
        << "The first dimension of `q_indices` must be equal to the "
           "first dimension of `centroids`.";

    TORCH_CHECK_EQ(centroids.sizes(), residual_centroids.value().sizes())
        << "The numel of centroids and residual_centroids must be the same.";

    TORCH_CHECK_EQ(q_indice_residual.value().device().index(), dev_index)
        << "the residual index tensor is on a different device.";
    TORCH_CHECK_EQ(centroids.device().index(), dev_index)
        << "the centroids tensor is on a different device.";
    TORCH_CHECK_EQ(residual_centroids.value().device().index(), dev_index)
        << "the residual centroids tensor is on a different device.";
    TORCH_CHECK_EQ(perm_dev_index, dev_index)
        << "the permutation index tensor is on a different device.";
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;
  const int64_t out_f_x_in_f[2] = {out_features, in_features};

  output = launch_deqantize_outliers_cuda_packkernel(
      out_f_x_in_f, q_indice, centroids, q_indice_residual, residual_centroids,
      q_indice_outliers, outliers_centroids, invperm, weight_scale,
      weight_bias);

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

}  // namespace vptq
