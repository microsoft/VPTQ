// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "copy/mod.cuh"

#include <cute/tensor.hpp>

namespace vptq::kernels {
using namespace cute;

namespace {
template <typename DType, const int kTileSize, const int kVecLen,
          const int kNumCentroids, const int kNumResCentroids>
struct SharedStorageImpl {
  ///==== Shared memory for inputs ====///
  static constexpr int kSizeCodebook = kNumCentroids * kVecLen;
  array_aligned<DType, kSizeCodebook, 128> codebook;  // 128-bits aligned

  static constexpr int kSizeCodebookRes = kNumResCentroids * kVecLen;
  array_aligned<DType, kSizeCodebookRes, 128> codebook_res;  // 128-bits aligned

  static constexpr int kSizeInputs = 3 * kTileSize;
  array_aligned<DType, kSizeInputs, 128> inputs;

  static constexpr int kSizeIndices = kTileSize * 2;
  array_aligned<uint16_t, kTileSize * 2> indices;

  ///==== Shared mempory for intermediate results ====///
  static constexpr int kSizeWeights = kTileSize * kVecLen;
  array_aligned<DType, kSizeWeights, 128> dequant_weights;

  static constexpr int kSizeOut = 2 * kVecLen;
  array_aligned<DType, kSizeOut> output;

  static constexpr int kSmemSize = ((kSizeCodebook + kSizeCodebookRes +
                                     kSizeInputs + kSizeWeights + kSizeOut) *
                                    sizeof(DType)) +
                                   kSizeIndices * sizeof(uint16_t);
};

template <typename DType, const int kThreads, const int kNumCentroids,
          const int kVecLen, typename Base = copy::AccessInfo<DType>>
struct CodebookTraits : public Base {
  static constexpr int kVecInBytes = kVecLen * sizeof(DType);
  // TODO: To support a vector length of 12, this constraint can be relaxed.
  static_assert(Base::kCacheLineBytes % kVecInBytes == 0,
                "The cache line size must be divisible by the vector size.");

  // The original codebook is reshaped into a matrix where the columns span
  // the entire shared memory banks.
  static constexpr int kPackedVecs = Base::kCacheLineBytes / kVecInBytes;
  static_assert(kNumCentroids % kPackedVecs == 0,
                "Current implementations require the number of centroids must "
                "be divisible by the number of packed vectors.");

  static constexpr int kCols = Base::kCacheLineBytes / sizeof(DType);
  static constexpr int kRows = kNumCentroids / kPackedVecs;
  static constexpr int kNumel = kVecLen * kNumCentroids;

  static constexpr int kThreadCols = kCols / Base::kNumPerAccess;
  static constexpr int kThreadRows = kThreads / kThreadCols;
  using ThreadLayout = cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                                    Stride<Int<kThreadCols>, _1>>;
  using Layout =
      cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
  using Loader = copy::GlobalToSharedLoader<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;

  // Storer is defined for debugging purposes
  using Storer = copy::SharedToGlobalStorer<DType, Base::kNumPerAccess,
                                            ThreadLayout, Layout, Layout>;
};

template <typename DType, const int kThreads, const int kTileSize,
          typename Base = copy::AccessInfo<DType>>
struct InputTraitsImpl : public Base {
  static constexpr int kThreadsTotal =
      kTileSize * sizeof(DType) / Base::kAccessInBytes;

  static_assert(kThreadsTotal <= kThreads,
                "The current implementation requires that the number of "
                "threads used to load a single input tile must be less than or "
                "equal to the number of threads in the block.");

  static constexpr int kRows = 1;
  static constexpr int kCols = kTileSize;

  static constexpr int kThreadRows = 1;
  static constexpr int kThreadCols = kThreadsTotal;

  using DataLayout =
      cute::Layout<Shape<_1, Int<kTileSize>>, Stride<Int<kTileSize>, _1>>;
  using ThreadLayout = cute::Layout<Shape<_1, Int<kThreadsTotal>>,
                                    Stride<Int<kThreadsTotal>, _1>>;
  using Loader =
      copy::GlobalToSharedLoader<DType, Base::kNumPerAccess, ThreadLayout,
                                 DataLayout, DataLayout>;
  // storer is defined for debugging purposes
  using Storer =
      copy::SharedToGlobalStorer<DType, Base::kNumPerAccess, ThreadLayout,
                                 DataLayout, DataLayout>;
};
}  // namespace

template <typename DType, const int kThreads,        //
          const int kTileSize_, const int kVecLen_,  //
          const int kNumCentroids_, const int kNumResCentroids_,
          typename Base = copy::AccessInfo<DType>>
struct QuantGemvKeTraits : public Base {
  static constexpr int kVecLen = kVecLen_;
  static constexpr int kNumCentroids = kNumCentroids_;
  static constexpr int kNumResCentroids = kNumResCentroids_;
  static constexpr int kTileSize = kTileSize_;

  // allocate shared memory
  using SharedStorage = SharedStorageImpl<DType, kTileSize, kVecLen,
                                          kNumCentroids, kNumResCentroids>;
  static constexpr int kSmemSize = SharedStorage::kSmemSize;

  using MainCentroidTraits =
      CodebookTraits<DType, kThreads, kNumCentroids, kVecLen>;
  using ResCentroidTraits =
      CodebookTraits<DType, kThreads, kNumResCentroids, kVecLen>;

  using InputTraits = InputTraitsImpl<DType, kThreads, kTileSize>;
  using InputLoader = typename InputTraits::Loader;
  using InputStorer = typename InputTraits::Storer;
};

}  // namespace vptq::kernels
