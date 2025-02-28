// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::tile_layout {

enum class Layout { kRowMajor = 0, kColMajor = 1 };

template <const int kRows_, const int kCols_, const int kRowStride_,
          const int kColStride_>
struct MatrixLayout {
  static constexpr int kRows = kRows_;
  static constexpr int kCols = kCols_;

  static constexpr int kRowStride = kRowStride_;
  static constexpr int kColStride = kColStride_;

  static constexpr int kNumel = kRows * kCols;

  // FIXME(ying): The current method to determine if the layout is row-major or
  // column-major may not be accurate for a matrix of shape (1, 1).
  static constexpr Layout kType =
      kColStride == 1 ? Layout::kRowMajor : Layout::kColMajor;

  HOST_DEVICE int operator()(int i, int j) const {
    return i * kRowStride + j * kColStride;
  }
};

template <const int kRow, const int kCol, const int kStride = kCol>
using RowMajor = MatrixLayout<kRow, kCol, kStride, 1>;
template <const int kRow, const int kCol, const int kStride = kRow>
using ColMajor = MatrixLayout<kRow, kCol, 1, kStride>;

}  // namespace vptq::tile_layout
