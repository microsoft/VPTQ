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
    const DType* const __restrict__ scale_bias,  //
    int64_t batch, int64_t seq_length,           //
    int64_t in_features, int64_t out_features) {
  extern __shared__ unsigned char buf_[];
  using SharedStorage = typename KeTraits::SharedStorage;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(buf_);

  static constexpr int kTileSize = KeTraits::kTileSize;

  // load the main centroids into shared memory
  typename KeTraits::MainCentroidTraits::Loader loader;
  loader(centroids, smem.codebook.data());

  if (residual_centroids) {
    // load the residual centroids into shared memory if available
    typename KeTraits::ResCentroidTraits::Loader loader;
    loader(residual_centroids, smem.codebook_res.data());
  }
  __copy_async();
  __syncthreads();

  // for debug
  // typename KeTraits::MainCentroidTraits::Storer storer;
  // storer(smem.codebook.data(), output);

  // typename KeTraits::ResCentroidTraits::Storer storer;
  // storer(smem.codebook_res.data(), output);

  typename KeTraits::InputLoader input_loader;
  typename KeTraits::InputStorer storer;

  // TODO(ying): naive implementation, need to be improved
  // Advance the input pointer to the current CTA
  int batch_id = blockIdx.x / seq_length;
  int seq_id = blockIdx.x % seq_length;

  const DType* x =
      batch_id * (seq_length * in_features) + seq_id * in_features + act;

  // for debug
  DType* y =
      batch_id * (seq_length * in_features) + seq_id * in_features + output;

  DType* s_input = smem.inputs.data();
  DType* s_scale_weights = scale_weights ? s_input + kTileSize : nullptr;
  DType* s_scale_bias = scale_bias ? s_scale_weights + kTileSize : nullptr;

  for (int k = 0; k < in_features; k += kTileSize) {
    input_loader(x + k, s_input + k);

    if (scale_weights) {
      input_loader(scale_weights + k, s_scale_weights + k);
      input_loader(scale_bias + k, s_scale_bias + k);
    }

    if (bias) {
    }
    __copy_async();
    __syncthreads();

    storer(s_input + k, y + k);
    __copy_async();
    __syncthreads();

    // decode

    // in-place scaling

    // dot product and add

    // store results for the current tile
  }

  return;
}

}  // namespace vptq::kernels
