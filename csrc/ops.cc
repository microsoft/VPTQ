// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>

namespace vptq {
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor launch_deqantize_outliers_cuda_packkernel(
    const int64_t* outf_x_inf, const torch::Tensor& q_indice,
    const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& outliers_indices,
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias);

torch::Tensor launch_gemv_outliers_cuda_packkernel(
    const int64_t out_features, const torch::Tensor& input,
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
                        const c10::optional<torch::Tensor>& bias,
                        int64_t groupsize, int64_t in_features,
                        int64_t out_features) {
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

  TORCH_CHECK_GE(groupsize, 2) << "groupsize must be >= 2.";

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

// NOTE: DO NOT change the module name "libvptq" here. It must match how
// the module is loaded in the Python codes.
PYBIND11_MODULE(libvptq, m) {
  m.doc() = "VPTQ customized kernels.";

  m.def("dequant", &vptq::dequant, "vptq customized dequantization kernel.");
  m.def("gemm", &vptq::wqA16Gemm, "vptq customized dequantized gemv kernel.");
}
