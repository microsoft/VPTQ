// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "dispatch_macros.h"
#include "kernels/quant_gemv_traits.cuh"
#include "kernels/quant_gemv_v2.cuh"
#include "util/common.h"

namespace vptq::kernels {

using namespace copy;

template <typename DType, typename KeTraits>
__global__ void ke_quant_gemv_v2(
    DType* __restrict__ output, const DType* const __restrict__ act,
    const DType* const __restrict__ bias,
    const int32_t* const __restrict__ indices,
    const DType* const __restrict__ centroids,
    const DType* const __restrict__ residual_centroids,
    const DType* const __restrict__ scale_weights,
    const DType* const __restrict__ scale_bias, int64_t in_features,
    int64_t out_features, int64_t vec_len) {
  extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
  DType* codebook1 = reinterpret_cast<DType*>(buf_);
  DType* codebook2 = codebook1 + KeTraits::MainCentroidTraits::kNumel;

  // load the main centroids into shared memory
  typename KeTraits::MainCentroidTraits::Loader loader;
  loader(centroids, codebook1);

  if (residual_centroids) {
    // load the residual centroids into shared memory if available
    typename KeTraits::ResCentroidTraits::Loader loader;
    loader(residual_centroids, codebook2);
  }
  __copy_async();
  __syncthreads();

  typename KeTraits::ResCentroidTraits::Storer storer;
  storer(codebook2, output);
  return;
}

}  // namespace vptq::kernels
