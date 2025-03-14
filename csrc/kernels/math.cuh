// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/reduce.cuh"

namespace vptq::kernels {

template <typename DType>
struct Sum {
  HOST_DEVICE DType operator()(const DType& a, const DType& b) const {
    return a + b;
  }
};

/// @param kNum: number of indices to decode per thread
/// @param kVecLen: vector length of the codebook
/// @param kPackedNums: how many floating-point numbers are packed in a
///                     single vector when accessing the codebook
template <typename IdLoader, typename ResIdLoader,   //
          typename ScaleLoader, typename VecLoader,  //
          typename Reducer,                          //
          const int kNum, const int kVecLen, const int kPackedNums>
struct GemvImpl {
  /// all pointers, except for init_vals, are shared memory pointers
  template <typename DType, typename IdType, typename ResIdType>
  DEVICE void operator()(
      DType* acc,                       // accumulated values
      const DType* input,               // input
      const IdType* main_idx,           // indices for main centroids
      const DType* main_codebook_,      // main codebook
      const ResIdType* res_idx_,        // indices for residual centroids
      const DType* res_codebook_,       // residual codebook
      const DType* scale,               // scale
      const DType* bias,                // bias
      IdType* idx, ResIdType* res_idx,  // indices
      DType* xs, DType* ss, DType* bs,  // input, scale, bias
      DType* vec, DType* res_vec) {     // dequantized weights
#if defined(__CUDA_ARCH__)
    ///===== 1. load input data from shared memory to registers =====///
    id_loader(main_idx, idx);
    res_id_loader(res_idx_, res_idx);

    scale_loader(input, xs);
    scale_loader(scale, ss);
    scale_loader(bias, bs);

    ///===== 2. decode the codebook vectors =====///
  #pragma unroll
    for (int i = 0; i < kNum; ++i) {
      const DType* main_codebook = main_codebook_ + idx[i] * kVecLen;
      const DType* res_codebook = res_codebook_ + res_idx[i] * kVecLen;

  #pragma unroll
      for (int j = 0; j < kVecLen; j += kPackedNums) {
        vec_loader(main_codebook + j, vec + j);
        vec_loader(res_codebook + j, res_vec + j);
      }

  #pragma unroll
      for (int j = 0; j < kVecLen; ++j) {
        // TODO(ying): Replace with vectorized operation
        vec[j] = xs[i] * (ss[i] * (vec[j] + res_vec[j]) + bs[i]);
      }

      /// warp reduction for dot product
      // shared memory to store intermediate results for warp reduction
      DType val;
      __shared__ DType shm[WARP_SIZE];
  #pragma unroll
      for (int j = 0; j < kVecLen; ++j) {
        val = vec[j];
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
