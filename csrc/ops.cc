
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor lauch_deqantize_outliers_cuda_packkernel(
    const int* outf_x_inf,                                   //[out_f, in_f]
    const torch::Tensor& q_indice,                           //[num_cen, o_c_size, in_inf]
    const torch::Tensor& centroids,                          //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>& q_indice_residual,   //[num_cen, o_c_size, in_inf]
    const c10::optional<torch::Tensor>& residual_centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>& outliers_centroids,  //[num_c, c_size, out_vec_len]
    const c10::optional<torch::Tensor>& outliers_indices,    //[num_cen, c_size, ol_in_f]
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale, const torch::Tensor& weight_bias);
torch::Tensor lauch_gemv_outliers_cuda_packkernel(
    const int out_features, const torch::Tensor& input,
    const torch::Tensor& q_indice,                           //[num_cen, o_c_size, in_inf]
    const torch::Tensor& centroids,                          //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>& q_indice_residual,   //[num_cen, o_c_size, in_inf]
    const c10::optional<torch::Tensor>& residual_centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>& outliers_indices,    //[num_cen, c_size, ol_in_f]
    const c10::optional<torch::Tensor>& outliers_centroids,  //[num_c, c_size, out_vec_len]
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale, const torch::Tensor& weight_bias,
    const c10::optional<torch::Tensor>& bias);

torch::Tensor dequant(const torch::Tensor& q_indice, const torch::Tensor& centroids,
                      const c10::optional<torch::Tensor>& q_indice_residual,
                      const c10::optional<torch::Tensor>& residual_centroids,
                      const c10::optional<torch::Tensor>& q_indice_outliers,
                      const c10::optional<torch::Tensor>& outliers_centroids,
                      const c10::optional<torch::Tensor>& invperm, const torch::Tensor& weight_scale,
                      const torch::Tensor& weight_bias, int groupsize, int in_features, int out_features) {
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
  TORCH_CHECK(q_indice.dtype() == torch::kInt, "q_indice must be int");
  TORCH_CHECK(groupsize >= 2, "groupsize must be >= 4");
  TORCH_CHECK(q_indice.dim() == 3, "must be 3D tensor");

  if (q_indice_residual.has_value()) {
    TORCH_CHECK(q_indice.size(0) == centroids.size(0) && centroids.sizes() == residual_centroids.value().sizes(),
                "size must be same");
    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index && centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index && perm_dev_index == dev_index,
                "all input tensors must be on the same device");
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;  // = at::empty({out_features, in_features}, centroids.options());
  const int out_f_x_in_f[2] = {out_features, in_features};

  output = lauch_deqantize_outliers_cuda_packkernel(out_f_x_in_f, q_indice, centroids, q_indice_residual,
                                                    residual_centroids, q_indice_outliers, outliers_centroids, invperm,
                                                    weight_scale, weight_bias);

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

torch::Tensor wqA16Gemm(const torch::Tensor& input, const torch::Tensor& q_indice, const torch::Tensor& centroids,
                        const c10::optional<torch::Tensor>& q_indice_residual,
                        const c10::optional<torch::Tensor>& residual_centroids,
                        const c10::optional<torch::Tensor>& q_indice_outliers,
                        const c10::optional<torch::Tensor>& outliers_centroids,
                        const c10::optional<torch::Tensor>& invperm, const torch::Tensor& weight_scale,
                        const torch::Tensor& weight_bias, const c10::optional<torch::Tensor>& bias, int groupsize,
                        int in_features, int out_features) {
  CHECK_INPUT(q_indice);
  CHECK_INPUT(input);
  if (q_indice_residual.has_value()) {
    CHECK_INPUT(q_indice_residual.value());
    CHECK_INPUT(residual_centroids.value());
  }
  TORCH_CHECK(q_indice.dtype() == torch::kInt, "q_indice must be int");

  CHECK_INPUT(centroids);
  auto dev_index = q_indice.device().index();
  auto inv_perm_device_index = dev_index;
  if (invperm.has_value()) {
    CHECK_INPUT(invperm.value());
    inv_perm_device_index = invperm.value().device().index();
  }
  TORCH_CHECK(groupsize >= 2, "groupsize must be >= 2");
  if (q_indice_residual.has_value()) {
    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index && centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index && inv_perm_device_index == dev_index,
                "all input tensors must be on the same device");
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;

  output = lauch_gemv_outliers_cuda_packkernel(out_features, input, q_indice, centroids, q_indice_residual,
                                               residual_centroids, q_indice_outliers, outliers_centroids, invperm,
                                               weight_scale, weight_bias, bias);

  gpuErrchk(cudaPeekAtLastError());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant", &dequant,
        "dequantize qweight to fp16, \nfunction type: const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, tensor g_idx, int groupsize, int bits, int "
        "in_features");
  m.def("gemm", &wqA16Gemm,
        "compute the gemm output, usually gemv, \nfunction type: const torch::Tensor& qweight, "
        "const torch::Tensor& scales, const torch::Tensor& qzeros, tensor g_idx, int groupsize, int bits, int "
        "in_features");
}
