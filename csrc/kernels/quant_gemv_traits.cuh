// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/copy/mod.cuh"
#include "kernels/decode.cuh"
#include "kernels/reduce.cuh"

#include <cute/tensor.hpp>

namespace vptq::kernels {
namespace tl = vptq::tile_layout;
using namespace cute;

namespace {  /// functions, structs, and constants that are not intend to expose
             /// to the global scope

template <typename DType, typename IdType, typename ResIdType,
          const int kTileSize, const int kVecLen, const int kNumCentroids,
          const int kNumResCentroids>
struct SharedStorageImpl {
  ///==== Shared memory for inputs ====///
  static constexpr int kSizeCodebook = kNumCentroids * kVecLen;
  array_aligned<DType, kSizeCodebook, 128> codebook;  // 128-bits aligned

  static constexpr int kSizeCodebookRes = kNumResCentroids * kVecLen;
  array_aligned<DType, kSizeCodebookRes, 128> codebook_res;  // 128-bits aligned

  // 3 stands for input tile, scale and bias.
  static constexpr int kSizeInputs = 3 * kTileSize;
  array_aligned<DType, kSizeInputs, 128> inputs;

  array_aligned<IdType, kTileSize> indices;         // for main centroids
  array_aligned<ResIdType, kTileSize> res_indices;  // for residual centroids

  // 2 stands for output and bias applied to the output.
  static constexpr int kSizeOut = 2 * kVecLen;
  array_aligned<DType, kSizeOut> output;

  static constexpr int kSmemSize =
      (kSizeCodebook + kSizeCodebookRes + kSizeInputs + kSizeOut) *
          sizeof(DType) +
      kTileSize * sizeof(IdType) + kTileSize * sizeof(ResIdType);
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
  ///===== constants =====///
  static constexpr int kVecLen = kVecLen_;
  static constexpr int kNumCentroids = kNumCentroids_;
  static constexpr int kNumResCentroids = kNumResCentroids_;
  static constexpr int kTileSize = kTileSize_;
  static constexpr int kNumWarps = kThreads / WARP_SIZE;

  // Number of warps required to load a single input tile. This may be fewer
  // than the total warps used in a CTA.
  static constexpr int kNumWarpsPerTile =
      kTileSize * sizeof(DType) / Base::kAccessInBytes / WARP_SIZE;
  static_assert(kNumWarps % kNumWarpsPerTile == 0,
                "The number of warps must be divisible by the number of warps "
                "used to load a single tile.");
  static constexpr int kWarpPerResIdsTile =
      kTileSize * sizeof(ResIdType) / Base::kAccessInBytes / WARP_SIZE;
  using WarpCounter = copy::WarpCounter<kNumWarps, kNumWarpsPerTile>;

  // Determines the number of threads needed to load a single index tile.
  // Since indices are stored as low-bit integers (e.g., uint16_t or uint8_t),
  // and each thread uses maximally vectorized memory instructions,
  // we can efficiently load the entire tile without requiring all threads
  // in the thread block to participate.
  static constexpr int kThreadsIndex =
      kTileSize * sizeof(IdType) / Base::kAccessInBytes;
  static_assert(kThreadsIndex <= kThreads && kThreadsIndex % WARP_SIZE == 0,
                "The number of threads required to load a single index tile "
                "must not exceed the total number of threads in the block.");

  static constexpr int kThreadsInput =
      kTileSize * sizeof(DType) / Base::kAccessInBytes;
  static_assert(kThreadsInput <= kThreads,
                "The number of threads required to load a single input tile "
                "must not exceed the total number of threads in the block.");

  static constexpr int kThreadsResIndex =
      kTileSize * sizeof(ResIdType) / Base::kAccessInBytes;
  static_assert(kThreadsResIndex <= kThreads &&
                    kThreadsResIndex % WARP_SIZE == 0,
                "The number of threads required to load a single residual "
                "index tile must not exceed the total number of threads in "
                "the block.");

  // Ensures indices are aligned with shared memory banks and each thread
  // decodes at least kIdsPerBank indices
  static constexpr int kBankBytes = 4;  // 4 bytes per bank
  // NOTE: We assume residual indices use the smallest bit-width data type among
  // all inputs (including FP8 data inputs). This assumption is critical for
  // memory alignment and access patterns. When modifying index data types,
  // verify this assumption still holds to avoid potential memory access issues.
  static_assert(kBankBytes % sizeof(ResIdType) == 0);
  static constexpr int kIdsPerBank = kBankBytes / sizeof(ResIdType);

  // Specifies how many indices each thread decodes, which determines
  // the number of codebook lookups performed by a single thread
  static_assert(kTileSize % (kThreads * kIdsPerBank) == 0);
  // how many indices are dequantized per thread
  static constexpr int kDequantNumPerThread = kTileSize / kThreads;

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

  // Determines how many floating-point numbers are packed in a single memory
  // access when loading from the codebook into registers.
  static constexpr int kPackedNums = kVecBytes > kAccessInBytes
                                         ? kAccessInBytes / sizeof(DType)
                                         : kVecLen;

  ///===== allocate shared memory =====///
  using SharedStorage =
      SharedStorageImpl<DType, IdType, ResIdType, kTileSize, kVecLen,
                        kNumCentroids, kNumResCentroids>;

  ///===== configurations for loading codebooks =====///
  using MainCentroidTraits =
      CodebookTraits<DType, kThreads, kNumCentroids, kVecLen>;
  using ResCentroidTraits =
      CodebookTraits<DType, kThreads, kNumResCentroids, kVecLen>;

  ///===== configurations for loading bias =====///
  using BiasLoader = copy::GlobalToSharedBiasLoader<DType, kVecLen>;
  // Storer is defined for debugging purposes only
  using BiasStorer = copy::SharedToGlobalBiasStorer<DType, kVecLen>;

  ///===== configurations for loading tiled input =====///
  using InputLoader = copy::GlobalToSharedInputLoader<DType, kTileSize>;
  // Storer is defined for debugging purposes only
  using InputStorer = copy::SharedToGlobalInputStorer<DType, kTileSize>;

  ///===== configurations for loading tiled indices =====///
  using IdLoader = copy::GlobalToSharedInputLoader<IdType, kTileSize>;
  using IdStorer = copy::SharedToGlobalInputStorer<IdType, kTileSize>;

  // loading tiled residual indices which may have different data type
  // (like uint8_t) from the main indices (like uint16_t)
  using ResIdLoader = copy::GlobalToSharedInputLoader<ResIdType, kTileSize>;
  using ResIdStorer = copy::SharedToGlobalInputStorer<ResIdType, kTileSize>;

  ///===== dequantizing weights and computing gemv on registers =====///
  using IdLoaderS2R = copy::PackedCopy<IdType, kDequantNumPerThread>;
  using ResIdLoaderS2R = copy::PackedCopy<ResIdType, kDequantNumPerThread>;
  using ScaleLoaderS2R = copy::PackedCopy<DType, kDequantNumPerThread>;

  using VecLoaderS2R = copy::PackedCopy<DType, kPackedNums>;
  using VecStorer = copy::PackedCopy<DType, kPackedNums>;

  // float is used for accumulating intermediate results.
  // DO NOT change to DType.
  using Reducer = Sum<float>;
  using Decode =
      DecodeImpl<IdLoaderS2R, ResIdLoaderS2R, ScaleLoaderS2R, VecLoaderS2R,
                 kDequantNumPerThread, kVecLen, kPackedNums>;
};

}  // namespace vptq::kernels
