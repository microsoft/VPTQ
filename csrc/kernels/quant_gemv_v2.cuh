// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "dispatch_macros.h"
#include "kernels/convert.cuh"
#include "kernels/copy/mod.cuh"
#include "kernels/quant_gemv_traits.cuh"
#include "util/common.h"
#include "util/cuda_utils.cuh"
#include "util/debug.cuh"

namespace vptq::kernels {
using namespace copy;

template <typename DType, typename IdType, typename ResIdType,
          typename SharedStorage, typename KeTraits>
__global__ void ke_quant_gemv_v2(DType* __restrict__ output,
                                 const DType* __restrict__ act,
                                 const DType* __restrict__ bias,
                                 const IdType* __restrict__ indices,
                                 const DType* __restrict__ centroids,
                                 const ResIdType* __restrict__ residual_indices,
                                 const DType* __restrict__ residual_centroids,
                                 const DType* __restrict__ scale_weights,
                                 const DType* __restrict__ scale_bias,
                                 int64_t batch, int64_t seq_length,
                                 int64_t in_features, int64_t out_features) {
  ///===== constants =====///
  static constexpr int kTileSize = KeTraits::kTileSize;
  static constexpr int kVecLen = KeTraits::kVecLen;
  static constexpr int kNumPerThread = KeTraits::kDequantNumPerThread;

  ///===== advance the pointers for data on global memory =====///
  int batch_id = blockIdx.x / seq_length;
  int seq_id = blockIdx.x % seq_length;
  const DType* x =  // activation
      act + batch_id * (seq_length * in_features) + seq_id * in_features;

  const IdType* g_ids = indices + blockIdx.z * in_features;
  const ResIdType* g_res_ids =
      residual_indices ? residual_indices + blockIdx.z * in_features : nullptr;

  const DType* g_bias = bias ? bias + blockIdx.z * KeTraits::kVecLen +
                                   threadIdx.x * KeTraits::kNumPerAccess
                             : nullptr;
  DType* y = output + batch_id * (seq_length * out_features) +
             seq_id * out_features + blockIdx.z * kVecLen;

  ///===== shared memory buffer =====///
  extern __shared__ unsigned char buf_[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(buf_);

  DType* s_output = smem.output.data();

  DType* s_codebook = smem.codebook.data();
  DType* s_codebook_res = smem.codebook_res.data();
  DType* s_inputs = smem.inputs.data();
  IdType* s_ids = smem.indices.data();
  ResIdType* s_res_ids = smem.res_indices.data();
  DType* s_scale_weights = scale_weights ? s_inputs + kTileSize : nullptr;
  DType* s_scale_bias = scale_bias ? s_scale_weights + kTileSize : nullptr;
  DType* s_bias = bias ? s_output + KeTraits::kVecLen : nullptr;

  ///===== 1. load data from global to shared memory =====///
  typename KeTraits::MainCentroidTraits::Loader loader;
  // load the main centroids into shared memory
  loader(centroids, s_codebook);
  if (residual_centroids) {
    // load the residual centroids into shared memory if available
    typename KeTraits::ResCentroidTraits::Loader loader;
    loader(residual_centroids, s_codebook_res);
  }
  __copy_async();
  __syncthreads();

  if (bias) {  // load the bias if available.
    typename KeTraits::BiasLoader loader;
    loader(g_bias, s_bias);
  }

  typename KeTraits::InputLoader input_loader;
  typename KeTraits::IdLoader id_loader;

  typename KeTraits::ResIdLoader res_id_loader;
  typename KeTraits::ResIdStorer res_id_storer;

  typename KeTraits::Gemv gemv;

  DType results[kVecLen];  // registers for accumulating results between tiles
  memset(results, 0, sizeof(DType) * kVecLen);

  typename KeTraits::WarpCounter counter;
  int wid = warpid();
  for (int step = 0; step < in_features; step += kTileSize) {  // over tiles
    counter.reset();
    if (wid < counter.next()) {
      // load a single tile of the activation into shared memory
      input_loader(x + step, s_inputs, counter.cur());
    }

    if (scale_weights) {
      ++counter;
      if (wid >= counter.cur() && wid < counter.next()) {
        // load scale_weights into shared memory if available
        input_loader(scale_weights + step, s_scale_weights, counter.cur());
      }

      ++counter;
      if (wid >= counter.cur() && wid < counter.next()) {
        // load scale_bias into shared memory if available
        input_loader(scale_bias + step, s_scale_bias, counter.cur());
      }
    }

    ++counter;
    if (wid >= counter.cur() && wid < counter.next()) {
      // load indices into shared memory
      id_loader(g_ids + step, s_ids, counter.cur());
    }

    if (residual_indices) {
      counter += KeTraits::kWarpPerResIdsTile;
      if (wid >= counter.cur() && wid < counter.next()) {
        // load residual indices into shared memory if available
        res_id_loader(g_res_ids + step, s_res_ids, counter.cur());
      }
    }
    __copy_async();
    __syncthreads();

    ///===== 2. decode, add residual, scale, dot product, accumulate results
    /// between tiles, apply bias on register =====///

    // advance the pointers to shared memory data for the current thread
    int offset = threadIdx.x * kNumPerThread;
    gemv(results,
         &s_inputs[offset],                   // input
         s_codebook, s_codebook_res,          // codebooks
         &s_ids[offset], &s_res_ids[offset],  // indices
         &s_scale_weights[offset], &s_scale_bias[offset]);
    __syncthreads();
  }

  ///===== 3. store final results =====///
  if (threadIdx.x == 0) {
    typename KeTraits::VecStorer storer;
    storer(results, s_output);  // store register tile to shared

    if (s_bias) {  // apply bias if available
      // FIXME(ying): replace with vectorized operation
      for (int i = 0; i < kVecLen; ++i) s_output[i] += s_bias[i];
    }

    for (int j = 0; j < kVecLen; j += KeTraits::kPackedNums)
      storer(s_output + j, y + j);  // store shared tile to global
  }
  return;
}

}  // namespace vptq::kernels
