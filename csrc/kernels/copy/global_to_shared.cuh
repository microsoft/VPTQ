// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

/// The Loader and Storer in this file use all collaborative threads in a thread
/// block to transfer data tiles between global memory and shared memory.

#include "kernels/copy/atom.cuh"
#include "kernels/copy/copy_traits.cuh"
#include "kernels/copy/warp.cuh"

#include <cute/tensor.hpp>

namespace {
template <const int a, const int b>
static constexpr int divup = (a + b - 1) / b;
}

namespace vptq::kernels::copy {
namespace tl = vptq::tile_layout;
using namespace cute;

template <typename DType, const int kNumPerAccess, typename ThreadLayout,
          typename GlobalLayout /*src*/, typename SharedLayout /*dst*/>
struct GlobalToSharedLoader {
  DEVICE void operator()(const DType* src_, DType* dst_) {
    int tid = threadIdx.x;

    auto gtile = make_tensor(make_gmem_ptr(src_), src_layout_);
    auto stile = make_tensor(make_smem_ptr(dst_), dst_layout_);

    auto loader = tiled_copy_.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
      for (int j = 0; j < int(size<2>(src)); ++j)
        cute::copy(tiled_copy_, src(cute::_, i, j), dst(cute::_, i, j));
  }

private:
  GlobalLayout src_layout_;
  SharedLayout dst_layout_;

#ifdef CP_ASYNC_SM80_ENABLED
  using CopyInst = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
  using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

  using TiledCopy =
      decltype(make_tiled_copy(CopyInst{}, ThreadLayout{},
                               cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));
  TiledCopy tiled_copy_;
};

template <typename DType, const int kNumPerAccess, typename ThreadLayout,
          typename SharedLayout /*src*/, typename GlobalLayout /*dst*/>
struct SharedToGlobalStorer {
  DEVICE void operator()(const DType* src_, DType* dst_) {
    int tid = threadIdx.x;

    auto stile = make_tensor(make_smem_ptr(src_), src_layout_);
    auto gtile = make_tensor(make_gmem_ptr(dst_), dst_layout_);

    auto loader = tiled_copy_.get_thread_slice(tid);

    auto src = loader.partition_S(stile);
    auto dst = loader.partition_D(gtile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
      for (int j = 0; j < int(size<2>(src)); ++j)
        cute::copy(tiled_copy_, src(cute::_, i, j), dst(cute::_, i, j));
  }

private:
  SharedLayout src_layout_;
  GlobalLayout dst_layout_;

  using TiledCopy =
      decltype(make_tiled_copy(Copy_Atom<DefaultCopy, DType>{}, ThreadLayout{},
                               cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));
  TiledCopy tiled_copy_;
};

/// NOTE: This configuration is specialized for copying a small tile whose size
/// is smaller than the data size accessed by all threads in a CTA concurrently.
template <typename DType, const int kNumel, typename Base = AccessInfo<DType>>
struct GlobalToSharedInputLoader : public Base {
  static constexpr int kWarpTileShape = Base::kNumPerAccess * WARP_SIZE;
  static constexpr int kThreads = kNumel / kWarpTileShape * WARP_SIZE;

  DEVICE void operator()(const DType* src_, DType* dst_, int start_warp = 0) {
    int warp_id = threadIdx.x / WARP_SIZE - start_warp;
    int lane_id = threadIdx.x % WARP_SIZE;
    int offset = warp_id * kWarpTileShape + lane_id * Base::kNumPerAccess;

    ld_global_st_shared<Base::kAccessInBytes>(
        __cvta_generic_to_shared(dst_ + offset), src_ + offset);
  }
};

/// NOTE: This configuration is specialized for copying a small tile whose size
/// is smaller than the data size accessed by all threads in a CTA concurrently.
template <typename DType, const int kNumel, typename Base = AccessInfo<DType>>
struct SharedToGlobalInputStorer : public Base {
  static constexpr int kWarpTileShape = Base::kNumPerAccess * WARP_SIZE;
  static constexpr int kThreads = kNumel / kWarpTileShape * WARP_SIZE;

  DEVICE void operator()(const DType* src_, DType* dst_, int start_warp = 0) {
    int warp_id = threadIdx.x / WARP_SIZE - start_warp;
    int lane_id = threadIdx.x % WARP_SIZE;
    int offset = warp_id * kWarpTileShape + lane_id * Base::kNumPerAccess;

    ld_shared_st_global<Base::kAccessInBytes>(
        dst_ + offset,
        static_cast<uint32_t>(__cvta_generic_to_shared(src_ + offset)));
  }
};

/// TODO(ying): all global to shared loaders and storers should be unified into
/// a single implementation. Manually writing each loader and storer according
/// to their different configurations is error-prone.

/// @brief Load bias from global memory to shared memory.
/// @param DType data type of the bias
/// @param kNumel number of elements to load
/// @param Base access info of the bias
template <typename DType, const int kNumel, typename Base = AccessInfo<DType>>
struct GlobalToSharedBiasLoader : public Base {
  static constexpr int kNumThreads =
      divup<kNumel * sizeof(DType), Base::kAccessInBytes>;

  DEVICE void operator()(const DType* src, DType* dst_) {
    if (threadIdx.x < kNumThreads) {
      uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_));
      ld_global_st_shared<16>(dst, src);
    }
  }
};

/// @brief Store bias from shared memory to global memory.
/// @param DType data type of the bias
/// @param kNumel number of elements to store
/// @param Base access info of the bias
template <typename DType, const int kNumel, typename Base = AccessInfo<DType>>
struct SharedToGlobalBiasStorer : public Base {
  static constexpr int kNumThreads =
      divup<kNumel * sizeof(DType), Base::kAccessInBytes>;

  DEVICE void operator()(const DType* src_, DType* dst) {
    if (threadIdx.x < kNumThreads) {
      uint32_t src = static_cast<uint32_t>(__cvta_generic_to_shared(src_));
      ld_shared_st_global<16>(dst, src);
    }
  }
};

}  // namespace vptq::kernels::copy
