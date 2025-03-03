// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/convert.cuh"  // for debug printing
#include "kernels/copy/copy_traits.cuh"
#include "kernels/copy/vectorized.cuh"
#include "kernels/math.cuh"
#include "kernels/reduce.cuh"
#include "util/debug.cuh"

namespace vptq::kernels {
using namespace copy;

template <typename DType_, typename IdType_, typename ResIdType_,
          const int kNumPerThread_, const int kVecLen_,
          typename Base = AccessInfo<DType_>>
struct WeightDecoder {
  using DType = DType_;
  using IdType = IdType_;
  using ResIdType = ResIdType_;

  // TODO(ying): The current implementation requires that the indices for both
  // main and residual centroids are stored in the same data type, such as both
  // being uint16_t. If the main indices are in uint16_t and the residual
  // indices are in uint8_t, additional handling will be required. This will be
  // addressed in the next version.
  static_assert(std::is_same_v<IdType, ResIdType>,
                "The data type of indices for main and residual centroids must "
                "be the same.");

  static constexpr int kNumPerThread = kNumPerThread_;
  static constexpr int kVecLen = kVecLen_;

  /// all pointers, except for init_vals, are shared memory pointers
  __device__ __forceinline__ void operator()(

      DType* out,                   // dequantized output
      const DType* init_vals,       // initial values
      const DType* input,           // input
      const DType* main_codebook_,  // main codebook
      const DType* res_codebook_,   // residual codebook
      const IdType* main_idx,       // indices for main centroids
      const ResIdType* res_idx,     // indices for residual centroids
      const DType* scale,           // scale
      const DType* bias) {          // bias
    /// Register storage for indices, scale/bias, and codebook vectors
    IdType reg_idx[kNumPerThread];
    ResIdType reg_res_idx[kNumPerThread];

    DType xs[kNumPerThread];
    DType ss[kNumPerThread];
    DType bs[kNumPerThread];

    DType reg_vec[kVecLen];
    DType reg_res_vec[kVecLen];

    /// Load indices and scale/bias to registers
    idx_loader(main_idx, reg_idx);
    idx_loader(res_idx, reg_res_idx);

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
        val = power2_reduce(val, shm, reducer, init_vals[j]);

        if (threadIdx.x == 0) reg_vec[j] = val;
      }

/// store the dequantized output from registers to shared memory
#pragma unroll
      for (int j = 0; j < kVecLen; j += kPackedNums) {
        vec_storer(reg_vec + j, out + i * kVecLen + j);
      }
    }
  }

private:
  Sum<DType> reducer;

  using IdxLoader = PackedCopy<IdType, kNumPerThread>;
  IdxLoader idx_loader;

  using ScaleLoader = PackedCopy<DType, kNumPerThread>;
  ScaleLoader scale_loader;

  // Here's an example to illustrate the vectorization constraint:
  // When a vector has length 16 and uses fp16/bf16 format, it occupies 256
  // bits (16 * 16 bits). Loading this would require two separate 128-bit
  // vectorized memory accesses.
  //
  // Two vectors may not be stored contiguously, which means they cannot
  // always be packed into a single vectorized access. For optimal
  // performance, each vector in the codebook must have a length that is a
  // multiple of the vectorization width (32-bit, 64-bit, or 128-bit).
  static constexpr int kVecBytes = sizeof(DType) * kVecLen;
  static constexpr int kAccessInBytes = Base::kAccessInBytes;
  static_assert(
      kVecBytes % kAccessInBytes == 0 || kAccessInBytes % kVecBytes == 0,
      "vector in the codebook must be aligned to the the width of the "
      "vectorization instruction.");

  // calculate how many numbers are packed within a single vector in the
  // codebook when accessing it from the codebook.
  static constexpr int kPackedNums = kVecBytes > kAccessInBytes
                                         ? kAccessInBytes / sizeof(DType)
                                         : kVecLen;

  using VecLoader = PackedCopy<DType, kPackedNums>;
  VecLoader vec_loader;

  using VecStorer = PackedCopy<DType, kPackedNums>;
  VecStorer vec_storer;
};

}  // namespace vptq::kernels
