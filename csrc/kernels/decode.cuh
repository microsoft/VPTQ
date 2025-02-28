// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/copy/copy_traits.cuh"
#include "kernels/copy/vectorized.cuh"
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

  DEVICE void operator()(DType* output,          // output
                         const DType* codebook,  // codebook for main centroids
                         const DType* codebook_res,  // codebook for residual
                         const IdType* ids,  // indices for main centroids
                         const ResIdType* res_ids,  // indices for residual
                         const DType* alpha, const DType* beta) {
    // threads in a CTA are laid out in 1-D fashion.
    int offset = threadIdx.x * kNumPerThread;
    const IdType* ids_ = ids + offset;  // indices for the current thread
    // residual indices for the current thread
    const ResIdType* res_ids_ = res_ids + offset;

    // load indices and residual indice into registers
    // indices on thread local registers
    IdType reg_ids[kNumPerThread];
    ResIdType reg_residual_ids[kNumPerThread];

#pragma unroll
    for (int i = 0; i < kNumPerThread; i += kPackedNum) {
      copy_ids(&ids_[i] /*src*/, &reg_ids[i] /*dst*/);
      copy_ids(&res_ids_[i] /*src*/, &reg_residual_ids[i] /*dst*/);
    }
  }

private:
  // Indices are packed into 4 bytes in the current implementation, stored in a
  // shared memory bank. This can be tuned if needed.
  static constexpr int kPackedIdsBytes = 4;
  static constexpr int kPackedNum = kPackedIdsBytes / sizeof(IdType);
  static_assert(kPackedNum, "kPackedNum must be greater than 0");
  using VecCopy = PackedCopy<IdType, kPackedNum>;
  VecCopy copy_ids;
};

}  // namespace vptq::kernels
