// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "util/cuda_utils.cuh"
#include "util/debug.cuh"

namespace vptq {

template <typename DType>
__global__ void quant_gemv_v2_kernel(
    DType* __restrict__ output, const DType* const __restrict__ act,
    const DType* const __restrict__ bias,
    const int32_t* const __restrict__ indices,
    const DType* const __restrict__ centroids,
    const DType* const __restrict__ residual_centroids,
    const DType* const __restrict__ scale_weights,
    const DType* const __restrict__ scale_bias, int64_t in_features,
    int64_t out_features, int64_t vec_len) {
  return;
}

}  // namespace vptq
