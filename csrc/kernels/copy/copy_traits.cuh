// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/copy/layout.cuh"

namespace vptq::kernels::copy {
namespace tl = vptq::tile_layout;

template <typename DType>
struct AccessInfo {
  // the maximal width of vectorized access.
  static constexpr int kAccessInBits = 128;
  static constexpr int kAccessInBytes = 16;

  static constexpr int kElementBits = sizeof(DType) * 8;
  static constexpr int kNumPerAccess = kAccessInBits / kElementBits;

  static constexpr int kCacheLineBytes = 128;  // 128 bytes
};

}  // namespace vptq::kernels::copy
