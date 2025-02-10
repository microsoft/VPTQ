// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "copy/copy_traits.cuh"

#include <cute/tensor.hpp>

namespace vptq::copy {
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

}  // namespace vptq::copy
