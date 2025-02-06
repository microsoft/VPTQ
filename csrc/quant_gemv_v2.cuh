// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "copy/sync.cuh"
#include "util/debug.cuh"

namespace vptq {
using namespace copy;

template <typename DType, typename KeTraits>
__global__ void quant_gemv_v2_kernel(
    DType* __restrict__ output, const DType* const __restrict__ act,
    const DType* const __restrict__ bias,
    const int32_t* const __restrict__ indices,
    const DType* const __restrict__ centroids,
    const DType* const __restrict__ residual_centroids,
    const DType* const __restrict__ scale_weights,
    const DType* const __restrict__ scale_bias, int64_t in_features,
    int64_t out_features, int64_t vec_len) {
  extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
  auto* buf = reinterpret_cast<DType*>(buf_);

  typename KeTraits::LoaderG2S loader;
  loader(centroids, buf);
  __copy_async();
  __syncthreads();

  return;
}

}  // namespace vptq
