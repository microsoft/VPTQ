// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/convert.cuh"
#include "kernels/reduce.cuh"
#include "util/debug.cuh"

namespace vptq::kernels {

/// @param kNum: number of indices to decode per thread
/// @param kVecLen: vector length of the codebook
/// @param kPackedNums: how many floating-point numbers are packed when
///                     accessing the codebook
template <typename IdLoader, typename ResIdLoader,   //
          typename ScaleLoader, typename VecLoader,  //
          const int kNum, const int kVecLen, const int kPackedNums>
struct DecodeImpl {
  /// all pointers, except for init_vals, are shared memory pointers
  template <typename DType, typename IdType, typename ResIdType>
  DEVICE void operator()(
      DType* acc,                       // accumulator between tiles
      const DType* s_input,             // input
      const IdType* s_idx,              // indices for main centroids
      const DType* s_codebook,          // main codebook
      const ResIdType* s_res_idx,       // indices for residual centroids
      const DType* s_res_codebook,      // residual codebook
      const DType* s_scale,             // scale
      const DType* s_bias,              // bias
      IdType* idx, ResIdType* res_idx,  // indices on registers
      DType* xs, DType* ss, DType* bs,  // input, scale, bias on registers
      DType* vec, DType* res_vec) {     // dequantized weights on registers

    id_loader(s_idx, idx);
    res_id_loader(s_res_idx, res_idx);

    scale_loader(s_input, xs);
    scale_loader(s_scale, ss);
    scale_loader(s_bias, bs);

    // TODO(ying): Apply vectorized operations to improve performance for these
    // nested loops. Currently, operations are performed element-wise without a
    // careful consideration of vectorization.
#pragma unroll
    for (int i = 0; i < kNum; ++i) {
      const DType* main_codebook = s_codebook + idx[i] * kVecLen;
      const DType* res_codebook = s_res_codebook + res_idx[i] * kVecLen;

#pragma unroll
      for (int j = 0; j < kVecLen; j += kPackedNums) {
        vec_loader(main_codebook + j, vec + j);
        vec_loader(res_codebook + j, res_vec + j);
      }
#pragma unroll
      for (int j = 0; j < kVecLen; ++j) {
        // TODO(ying): Replace with vectorized operations
        acc[j] += xs[i] * (ss[i] * (vec[j] + res_vec[j]) + bs[i]);
      }
    }
  }

private:
  IdLoader id_loader;
  ResIdLoader res_id_loader;
  ScaleLoader scale_loader;
  VecLoader vec_loader;
};

}  // namespace vptq::kernels
