// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

namespace vptq {

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
    const torch::Tensor& activations, const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& indices, const torch::Tensor& centroids,
    const c10::optional<torch::Tensor>& residual_centroids,
    const torch::Tensor& scale_weights, const torch::Tensor& scale_bias,
    int64_t in_features, int64_t out_features);

}  // namespace vptq
