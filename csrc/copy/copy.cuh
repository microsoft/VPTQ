// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "copy/copy_traits.cuh"

#include <cute/tensor.hpp>

namespace vptq::copy {
using namespace cute;

/// TODO(ying); the current implementation supports load row-major data only.
template <typename DType, const int kThreads, const int64_t kRows,
          const int64_t kCols, typename Base = AccessInfo<DType>>
struct GlobalToSharedLoader : public Base {
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
  // source
  using GlobalLayout =
      cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
  GlobalLayout src_layout_;

  // destination
  using SharedLayout =
      cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;

  // using LayoutAtom =
  //     decltype(composition(cute::Swizzle<2, 3, 3>{},
  //                          cute::Layout<Shape<_4, _64>, Stride<_64, _1>>{}));
  // using SharedLayout = decltype(tile_to_shape(
  //     LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));
  SharedLayout dst_layout_;

  // tiled copy
  static constexpr int kThreadCols =
      kCols * Base::kElementBits / Base::kAccessInBits;
  static_assert(kThreadCols > 0);
  static constexpr int kThreadRows = kThreads / kThreadCols;

  using ThreadLayout = cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                                    Stride<Int<kThreadCols>, _1>>;
  using ValueLayout = cute::Layout<Shape<_1, _8>>;

#ifdef CP_ASYNC_SM80_ENABLED
  using CopyInst = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
  using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

  using TiledCopy =
      decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));
  TiledCopy tiled_copy_;
};

}  // namespace vptq::copy
