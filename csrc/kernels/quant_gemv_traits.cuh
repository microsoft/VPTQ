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

  // 3 stands for input, scale_weights, scale_bias
  static constexpr int kSizeInputs = kTileSize * 3;
  array_aligned<DType, kSizeInputs> inputs;

  array_aligned<DType, kVecLen> bias;

  // TODO(ying): placeholder, a non-packed indices
  array_aligned<int32_t, kTileSize * 2> indices;

  ///==== Shared mempory for intermediate results ====///
  static constexpr int kSizeWeights = kTileSize * kVecLen;
  array_aligned<DType, kSizeWeights, 128> dequant_weights;

  ///==== Shared mempory for outputs ====///
  static constexpr int kSizeOut = kVecLen;
  array_aligned<DType, kSizeOut> output;

  static constexpr int kSmemSize =
      ((kSizeCodebook + kSizeCodebookRes + kSizeInputs + kVecLen +
        kSizeWeights + kSizeOut) *
       sizeof(DType)) +
      (kTileSize * 2 * sizeof(int32_t));
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

  // how many threads are used to load a single input tile;
  static constexpr int kInputThreads =
      kTileSize * sizeof(DType) / Base::kAccessInBytes;

  // Note: This is a basic implementation that currently may lead to performance
  // issues and needs optimization.
  static_assert(kInputThreads <= kThreads,
                "The current implementation requires that the number of "
                "threads used to load a single input tile must be less than or "
                "equal to the number of threads in the block.");
  using InputThreadLayout = cute::Layout<Shape<_1, Int<kInputThreads>>,
                                         Stride<Int<kInputThreads>, _1>>;
  using InputLayout =
      cute::Layout<Shape<_3, Int<kTileSize>>, Stride<Int<kTileSize>, _1>>;
  using InputLoader =
      copy::GlobalToSharedLoader<DType, Base::kNumPerAccess, InputThreadLayout,
                                 InputLayout, InputLayout>;

  // storer is defined for debugging purposes
  using InputStorer =
      copy::SharedToGlobalStorer<DType, Base::kNumPerAccess, InputThreadLayout,
                                 InputLayout, InputLayout>;
};

}  // namespace vptq::kernels
