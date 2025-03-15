// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// register bindings for VPTQ APIs in this file. ///
#include <torch/extension.h>

namespace vptq {

torch::Tensor dequant(const torch::Tensor& q_indice,
                      const torch::Tensor& centroids,
                      const c10::optional<torch::Tensor>& q_indice_residual,
                      const c10::optional<torch::Tensor>& residual_centroids,
                      const c10::optional<torch::Tensor>& q_indice_outliers,
                      const c10::optional<torch::Tensor>& outliers_centroids,
                      const c10::optional<torch::Tensor>& invperm,
                      const torch::Tensor& weight_scale,
                      const torch::Tensor& weight_bias, int64_t groupsize,
                      int64_t in_features, int64_t out_features);

torch::Tensor wquant_act16_gemv(
    const torch::Tensor& input, const torch::Tensor& q_indice,
    const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& q_indice_outliers,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& invperm,
    const torch::Tensor& weight_scale, const torch::Tensor& weight_bias,
    const c10::optional<torch::Tensor>& bias, int64_t in_features,
    int64_t out_features);

torch::Tensor quant_gemv_v2(
    const torch::Tensor& act, const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& indices, const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& residual_indices,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& scale_weights,
    const c10::optional<torch::Tensor>& scale_bias, int64_t out_features);

}  // namespace vptq

// NOTE: DO NOT change the module name "libvptq" here. It must match how
// the module is loaded in the Python codes.
PYBIND11_MODULE(libvptq, m) {
  m.doc() = "VPTQ customized kernels.";

  // v1 kernels.
  m.def("dequant", &vptq::dequant, "vptq customized dequantization kernel.");
  m.def("quant_gemv", &vptq::wquant_act16_gemv,
        "vptq customized quantized gemv kernel.");

  // v2 kernels.
  m.def("quant_gemv_v2", &vptq::quant_gemv_v2,
        "vptq customized quantized gemv kernel v2.");
}
