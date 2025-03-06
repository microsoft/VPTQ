// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/convert.cuh"  // for debug printing
#include "kernels/copy/mod.cuh"
#include "kernels/reduce.cuh"

#include <cute/tensor.hpp>

namespace vptq::kernels {
namespace tl = vptq::tile_layout;
using namespace cute;

namespace {

template <const int a, const int b>
static constexpr int divup = (a + b - 1) / b;

template <typename DType>
struct Sum {
  HOST_DEVICE DType operator()(const DType& a, const DType& b) const {
    return a + b;
  }
};

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

  static constexpr int kSizeOut = 2 * kVecLen;
  array_aligned<DType, kSizeOut> output;

  static constexpr int kSmemSize =
      (kSizeCodebook + kSizeCodebookRes + kSizeInputs + kSizeOut) *
          sizeof(DType) +
      kTileSize * sizeof(IdType) + kTileSize * sizeof(DType);
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

template <typename IndexLoader, typename ScaleLoader, typename VecLoader,
          typename Reducer, const int kNumPerThread, const int kVecLen,
          const int kPackedNums>
struct GemvImpl {
  /// all pointers, except for init_vals, are shared memory pointers
  template <typename DType, typename IdType, typename ResIdType>
  __device__ __forceinline__ void operator()(
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
    }
  }

private:
  Reducer reducer;
  IndexLoader idx_loader;
  ScaleLoader scale_loader;
  VecLoader vec_loader;
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

  // TODO(ying): The current implementation requires that the indices for both
  // main and residual centroids are stored in the same data type, such as both
  // being uint16_t. If the main indices are in uint16_t and the residual
  // indices are in uint8_t, additional handling will be required. This will be
  // addressed in the next version.
  static_assert(std::is_same_v<IdType, ResIdType>,
                "The data type of indices for main and residual centroids must "
                "be the same.");

  /// configurations for dequantizing weights and computing gemv on registers
  using IndexLoaderS2R = copy::PackedCopy<IdType, kDecodeNumPerThread>;
  using ScaleLoaderS2R = copy::PackedCopy<DType, kDecodeNumPerThread>;

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

  using VecLoaderS2R = copy::PackedCopy<DType, kPackedNums>;
  using VecStorer = copy::PackedCopy<DType, kPackedNums>;

  using Reducer = Sum<DType>;
  using Gemv = GemvImpl<IndexLoaderS2R, ScaleLoaderS2R, VecLoaderS2R, Reducer,
                        kDecodeNumPerThread, kVecLen, kPackedNums>;
};

}  // namespace vptq::kernels
