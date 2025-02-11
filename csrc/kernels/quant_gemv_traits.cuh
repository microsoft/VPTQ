// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "copy/mod.cuh"

#include <cute/tensor.hpp>

namespace vptq::kernels {

using namespace cute;

template <typename DType, const int kThreads, const int64_t kNumCentroids,
          const int64_t kVecLen, typename Base = copy::AccessInfo<DType>>
struct CodebookTraits : public Base {
  static constexpr int kVecInBytes = kVecLen * sizeof(DType);
  // TODO: To support a vector length of 12, this constraint can be relaxed.
  static_assert(Base::kCacheLineBytes % kVecInBytes == 0,
                "The cache line size must be divisible by the vector size.");

  static constexpr int kPackedVecs = Base::kCacheLineBytes / kVecInBytes;
  static_assert(kNumCentroids % kPackedVecs == 0,
                "Current implementations require the number of centroids must "
                "be divisible by the number of packed vectors.");

  static constexpr int kRows = kNumCentroids / kPackedVecs;
  static constexpr int kCols = kVecLen * kPackedVecs;
  static constexpr int kNumel = kRows * kCols;

  static constexpr int kThreadCols =  // how many threads are laid out in a row.
      kCols * Base::kElementBits / Base::kAccessInBits;
  static_assert(kThreadCols > 0);
  static constexpr int kThreadRows = kThreads / kThreadCols;

  using ThreadLayout = cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                                    Stride<Int<kThreadCols>, _1>>;
  using Layout =
      cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
  using Loader = copy::GlobalToSharedLoader<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;
  using Storer = copy::SharedToGlobalStorer<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;
};

template <typename DType, const int kThreads, const int64_t kNumCentroids,
          const int64_t kVecLen, typename Base = copy::AccessInfo<DType>>
struct ResCodebookTraits : public Base {
  static constexpr int kVecInBytes = kVecLen * sizeof(DType);
  // TODO: To support a vector length of 12, this constraint can be relaxed.
  static_assert(Base::kCacheLineBytes % kVecInBytes == 0,
                "The cache line size must be divisible by the vector size.");

  static constexpr int kPackedVecs = Base::kCacheLineBytes / kVecInBytes;
  static_assert(kNumCentroids % kPackedVecs == 0,
                "Current implementations require the number of centroids must "
                "be divisible by the number of packed vectors.");

  static constexpr int kRows = kNumCentroids / kPackedVecs;
  static constexpr int kCols = kVecLen * kPackedVecs;
  static constexpr int kNumel = kRows * kCols;

  static constexpr int kThreadCols =  // how many threads are laid out in a row.
      kCols * Base::kElementBits / Base::kAccessInBits;
  static_assert(kThreadCols > 0);

  static constexpr int kThreadsTotal =
      kNumel * sizeof(DType) * 8 / Base::kAccessInBits;

  static constexpr int kThreadRows = kThreadsTotal < kThreads
                                         ? kThreadsTotal / kThreadCols
                                         : kThreads / kThreadCols;

  using ThreadLayout = cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                                    Stride<Int<kThreadCols>, _1>>;
  using Layout =
      cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
  using Loader = copy::GlobalToSharedLoader<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;
  using Storer = copy::SharedToGlobalStorer<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;
};

template <typename DType, const int kThreads, const int64_t kNumCentroids,
          const int64_t kNumResCentroids, const int64_t kVecLen>
struct QuantGemvKeTraits {
  using MainCentroidTraits =
      CodebookTraits<DType, kThreads, kNumCentroids, kVecLen>;

  using ResCentroidTraits =
      ResCodebookTraits<DType, kThreads, kNumResCentroids, kVecLen>;
};

}  // namespace vptq::kernels
