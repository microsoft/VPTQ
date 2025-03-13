// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/reduce.cuh"

namespace vptq::kernels {

template <const int a, const int b>
static constexpr int divup = (a + b - 1) / b;

template <typename DType>
struct Sum {
  HOST_DEVICE DType operator()(const DType& a, const DType& b) const {
    return a + b;
  }
};

template <typename IdLoader, typename ResIdLoader,   //
          typename ScaleLoader, typename VecLoader,  //
          typename Reducer, const int kNumPerThread, const int kVecLen,
          const int kPackedNums>
struct GemvImpl {
  /// all pointers, except for init_vals, are shared memory pointers
  template <typename DType, typename IdType, typename ResIdType>
  DEVICE void operator()(
      DType* acc,                   // accumulated values
      const DType* input,           // input
      const DType* main_codebook_,  // main codebook
      const DType* res_codebook_,   // residual codebook
      const IdType* main_idx,       // indices for main centroids
      const ResIdType* res_idx,     // indices for residual centroids
      const DType* scale,           // scale
      const DType* bias) {          // bias
#if defined(__CUDA_ARCH__)
    /// Register storage for indices, scale/bias, and codebook vectors
    IdType reg_idx[kNumPerThread];
    ResIdType reg_res_idx[kNumPerThread];

    DType xs[kNumPerThread];
    DType ss[kNumPerThread];
    DType bs[kNumPerThread];

    DType reg_vec[kVecLen];
    DType reg_res_vec[kVecLen];

    /// Load indices and scale/bias to registers
    id_loader(main_idx, reg_idx);
    res_id_loader(res_idx, reg_res_idx);

    scale_loader(input, xs);
    scale_loader(scale, ss);
    scale_loader(bias, bs);

    // shared memory to store intermediate results for warp reduction
    DType val;
    __shared__ DType shm[WARP_SIZE];

    /// decode the codebook vectors
  #pragma unroll
    for (int i = 0; i < kNumPerThread; ++i) {
      const DType* main_codebook = main_codebook_ + reg_idx[i] * kVecLen;
      const DType* res_codebook = res_codebook_ + reg_res_idx[i] * kVecLen;

  #pragma unroll
      for (int j = 0; j < kVecLen; j += kPackedNums) {
        vec_loader(main_codebook + j, reg_vec + j);
        vec_loader(res_codebook + j, reg_res_vec + j);
      }

  #pragma unroll
      for (int j = 0; j < kVecLen; ++j) {
        // TODO(ying): Replace with vectorized operation
        reg_vec[j] = xs[i] * (ss[i] * (reg_vec[j] + reg_res_vec[j]) + bs[i]);
      }

      /// warp reduction for dot product
  #pragma unroll
      for (int j = 0; j < kVecLen; ++j) {
        val = reg_vec[j];
        val = power2_reduce(val, shm, reducer, static_cast<DType>(0));

        if (threadIdx.x == 0) acc[j] += val;
      }
    }
#else
    assert(false && "This function should only be called on the GPU.");
#endif
  }

private:
  Reducer reducer;
  IdLoader id_loader;
  ResIdLoader res_id_loader;
  ScaleLoader scale_loader;
  VecLoader vec_loader;
};

}  // namespace vptq::kernels
