// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/copy/mod.cuh"
#include "kernels/decode.cuh"

#include <cute/tensor.hpp>

namespace vptq::kernels {
namespace tl = vptq::tile_layout;

using namespace cute;

namespace {

template <const int a, const int b>
static constexpr int divup = (a + b - 1) / b;

template <typename DType, typename IdType, typename ResIdType,
          const int kTileSize, const int kVecLen, const int kNumCentroids,
          const int kNumResCentroids>
struct SharedStorageImpl {
  ///==== Shared memory for inputs ====///
  static constexpr int kSizeCodebook = kNumCentroids * kVecLen;
  array_aligned<DType, kSizeCodebook, 128> codebook;  // 128-bits aligned

  static constexpr int kSizeCodebookRes = kNumResCentroids * kVecLen;
  array_aligned<DType, kSizeCodebookRes, 128> codebook_res;  // 128-bits aligned

  static constexpr int kSizeInputs = 3 * kTileSize;
  array_aligned<DType, kSizeInputs, 128> inputs;

  // TODO(ying): Support residual indices are stored in uint8_t
  static_assert(std::is_same_v<IdType, ResIdType>,
                "The data type of indices for main and residual centroids must "
                "be the same.");
  array_aligned<IdType, kTileSize * 2> indices;

  ///==== Shared mempory for intermediate results ====///
  static constexpr int kSizeWeights = kTileSize * kVecLen;
  array_aligned<DType, kSizeWeights, 128> dequant_weights;

  static constexpr int kSizeOut = 2 * kVecLen;
  array_aligned<DType, kSizeOut> output;

  static constexpr int kSmemSize = ((kSizeCodebook + kSizeCodebookRes +
                                     kSizeInputs + kSizeWeights + kSizeOut) *
                                    sizeof(DType)) +
                                   2 * kTileSize * sizeof(IdType);
};

template <typename DType, const int kThreads, const int kNumCentroids,
          const int kVecLen, typename Base = copy::AccessInfo<DType>>
struct CodebookTraits : public Base {
  static_assert(kThreads % WARP_SIZE == 0,
                "The number of threads must be divisible by the warp size.");

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

template <typename DType, typename IdType, typename ResIdType,
          const int kThreads,                        //
          const int kTileSize_, const int kVecLen_,  //
          const int kNumCentroids_, const int kNumResCentroids_,
          typename Base = copy::AccessInfo<DType>>
struct QuantGemvKeTraits : public Base {
  /// constants
  static constexpr int kVecLen = kVecLen_;
  static constexpr int kNumCentroids = kNumCentroids_;
  static constexpr int kNumResCentroids = kNumResCentroids_;
  static constexpr int kTileSize = kTileSize_;

  /// allocate shared memory
  using SharedStorage =
      SharedStorageImpl<DType, IdType, ResIdType, kTileSize, kVecLen,
                        kNumCentroids, kNumResCentroids>;
  /// configurations for loading codebooks
  using MainCentroidTraits =
      CodebookTraits<DType, kThreads, kNumCentroids, kVecLen>;
  using ResCentroidTraits =
      CodebookTraits<DType, kThreads, kNumResCentroids, kVecLen>;

  /// configurations for loading bias
  static constexpr int kBiasLoadThreads =
      divup<kVecLen * sizeof(DType), Base::kAccessInBytes>;

  /// configurations for loading tiled input
  static constexpr int kNumWarps = kThreads / WARP_SIZE;
  // Number of warps required to load a single input tile. This may be fewer
  // than the total warps used in a CTA.
  static constexpr int kNumWarpsPerTile =
      kTileSize * sizeof(DType) / Base::kAccessInBytes / WARP_SIZE;
  static_assert(kNumWarps % kNumWarpsPerTile == 0,
                "The number of warps must be divisible by the number of warps "
                "used to load a single tile.");
  using WarpCounter = copy::WarpCounter<kNumWarps, kNumWarpsPerTile>;

  /// configurations for loading tiled input
  static constexpr int kThreadsInput =
      kTileSize * sizeof(DType) / Base::kAccessInBytes;
  static_assert(kThreadsInput <= kThreads,
                "The number of threads required to load a single input tile "
                "must not exceed the total number of threads in the block.");
  using InputLoader = copy::GlobalToSharedInputLoader<DType, kTileSize>;
  // Storer class defined for debugging purposes only
  using InputStorer = copy::SharedToGlobalInputStorer<DType, kTileSize>;

  /// configurations for loading tiled indices
  static constexpr int kThreadsIndex =
      kTileSize * sizeof(IdType) / Base::kAccessInBytes;
  static_assert(kThreadsIndex <= kThreads,
                "The number of threads required to load a single index tile "
                "must not exceed the total number of threads in the block.");

  // TODO(ying): Currently, indices for both main and residual centroids must
  // use the same data type. This limitation will be removed in the next
  // version.
  static_assert(std::is_same_v<IdType, ResIdType>,
                "The data type of indices for main and residual centroids must "
                "be the same.");
  using IndexLoader = copy::GlobalToSharedInputLoader<IdType, 2 * kTileSize>;
  using IndexStorer = copy::SharedToGlobalInputStorer<IdType, 2 * kTileSize>;

  /// configurations for decoding indices
  // Ensures indices are aligned with shared memory banks and each thread
  // decodes at least kIdsPerBank indices
  static constexpr int kBankBytes = 4;
  static_assert(kBankBytes % sizeof(ResIdType) == 0);
  static constexpr int kIdsPerBank = kBankBytes / sizeof(ResIdType);
  // Specifies how many indices each thread decodes, which determines
  // the number of codebook lookups performed by a single thread
  static_assert(kTileSize % (kThreads * kIdsPerBank) == 0);
  static constexpr int kDecodeNumPerThread = kTileSize / kThreads;

  using Decoder =
      WeightDecoder<DType, IdType, ResIdType, kDecodeNumPerThread, kVecLen>;
};

}  // namespace vptq::kernels
