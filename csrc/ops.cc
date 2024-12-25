// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// @brief dequantize qweight to fp16
// @param outf_x_inf has a shape of [output_feature, input_feature]
// @param q_indice has a shape of [num_centroid, output_channel_size, in_inf]
// @param centroids has a shape of [num_c, c_size, vec_len]
// @param q_indice_residual has a shape of [num_centroid, output_channel_size,
// @param residual_centroids has a shape of [num_c, c_size, vec_len]
// @param outliers_centroids has a shape of [num_c, c_size, out_vec_len]
// @param outliers_indices has a shape of [num_centroid, c_size, ol_in_f]
// @param perm has a shape of [num_centroid, c_size, ol_in_f]
// @param weight_scale has a shape of [output_feature, input_feature]
// @param weight_bias has a shape of [output_feature, input_feature]
torch::Tensor launch_deqantize_outliers_cuda_packkernel(
    const int* outf_x_inf, const torch::Tensor& q_indice,
    const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& outliers_indices,
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias);

// @brief compute the gemm output, usually gemv
// @param out_features,
// @param input,
// @param q_indice has a shape of [num_centroid, output_channel_size, in_inf]
// @param centroids has a shape of [num_c, c_size, vec_len]
// @param q_indice_residual has a shape of [num_centroid, output_channel_size,
// @param residual_centroids has a shape of [num_c, c_size, vec_len]
// @param outliers_indices has a shape of [num_centroid, c_size, ol_in_f]
// @param outliers_centroids has a shape of [num_c, c_size, out_vec_len]
// @param perm has a shape of [num_centroid, c_size, ol_in_f]
// @param weight_scale has a shape of [output_feature, input_feature]
// @param weight_bias has a shape of [output_feature, input_feature]
// @param bias has a shape of [output_feature]
// @return
torch::Tensor launch_gemv_outliers_cuda_packkernel(
    const int out_features, const torch::Tensor& input,
    const torch::Tensor& q_indice, const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& outliers_indices,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias, const c10::optional<torch::Tensor>& bias);

torch::Tensor dequant(const torch::Tensor& q_indice,
                      const torch::Tensor& centroids,
                      const c10::optional<torch::Tensor>& q_indice_residual,
                      const c10::optional<torch::Tensor>& residual_centroids,
                      const c10::optional<torch::Tensor>& q_indice_outliers,
                      const c10::optional<torch::Tensor>& outliers_centroids,
                      const c10::optional<torch::Tensor>& invperm,
                      const torch::Tensor& weight_scale,
                      const torch::Tensor& weight_bias, int groupsize,
                      int in_features, int out_features) {
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
  TORCH_CHECK(q_indice.dtype() == torch::kInt,
              "`q_indice` must have a type of integer.");
  TORCH_CHECK_GE(groupsize, 2) << "groupsize must be >= 4.";
  TORCH_CHECK_EQ(q_indice.dim(), 3) << "`q_indice` must be a 3D tensor.";

  if (q_indice_residual.has_value()) {
    TORCH_CHECK_EQ(q_indice.size(0), centroids.size(0))
        << "The first dimension of `q_indices` must be equal to the "
           "first dimension of `centroids`.";

    TORCH_CHECK_EQ(centroids.sizes(), residual_centroids.value().sizes())
        << "The numel of centroids and residual_centroids must be the same.";

    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index &&
                    centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index &&
                    perm_dev_index == dev_index,
                "All input tensors must be on the same device.");
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;
  const int out_f_x_in_f[2] = {out_features, in_features};

  output = launch_deqantize_outliers_cuda_packkernel(
      out_f_x_in_f, q_indice, centroids, q_indice_residual, residual_centroids,
      q_indice_outliers, outliers_centroids, invperm, weight_scale,
      weight_bias);

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

/// @brief compute the gemm output, usually gemv
/// @param input
/// @param q_indice
/// @param centroids
/// @param q_indice_residual
/// @param residual_centroids
/// @param q_indice_outliers
/// @param outliers_centroids
/// @param invperm
/// @param weight_scale
/// @param weight_bias
/// @param bias
/// @param groupsize
/// @param in_features
/// @param out_features
/// @return
torch::Tensor wqA16Gemm(const torch::Tensor& input,
                        const torch::Tensor& q_indice,
                        const torch::Tensor& centroids,
                        const c10::optional<torch::Tensor>& q_indice_residual,
                        const c10::optional<torch::Tensor>& residual_centroids,
                        const c10::optional<torch::Tensor>& q_indice_outliers,
                        const c10::optional<torch::Tensor>& outliers_centroids,
                        const c10::optional<torch::Tensor>& invperm,
                        const torch::Tensor& weight_scale,
                        const torch::Tensor& weight_bias,
                        const c10::optional<torch::Tensor>& bias, int groupsize,
                        int in_features, int out_features) {
  CHECK_INPUT(q_indice);
  CHECK_INPUT(input);
  if (q_indice_residual.has_value()) {
    CHECK_INPUT(q_indice_residual.value());
    CHECK_INPUT(residual_centroids.value());
  }
  TORCH_CHECK(q_indice.dtype() == torch::kInt, "q_indice must be integers.");

  CHECK_INPUT(centroids);
  auto dev_index = q_indice.device().index();
  auto inv_perm_device_index = dev_index;

  if (invperm.has_value()) {
    CHECK_INPUT(invperm.value());
    inv_perm_device_index = invperm.value().device().index();
  }

  TORCH_CHECK_GE(groupsize, 2) << "groupsize must be >= 2.";

  if (q_indice_residual.has_value()) {
    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index &&
                    centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index &&
                    inv_perm_device_index == dev_index,
                "All input tensors must be on the same device.");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant", &dequant,
        R"DOC(Dequantize matrix weights to fp16
function type:
const torch::Tensor& qweight,
const torch::Tensor& scales,
const torch::Tensor& qzeros,
Tensor g_idx, int groupsize, int bits, int in_features
)DOC");

  m.def("gemm", &wqA16Gemm,
        R"DOC(Compute the gemm output, usually gemv.
function type:
const torch::Tensor& qweight,
const torch::Tensor& scales,
const torch::Tensor& qzeros,
tensor g_idx, int groupsize, int bits, int in_features
)DOC");
}
