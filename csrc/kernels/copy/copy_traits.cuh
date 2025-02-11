// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels::copy {

template <typename DType>
struct AccessInfo {
  // the maximal width of vectorized access.
  static constexpr int kAccessInBits = 128;

  static constexpr int kElementBits = sizeof(DType) * 8;
  static constexpr int kNumPerAccess = kAccessInBits / kElementBits;

  static constexpr int kCacheLineBytes = 128;  // 128 bytes
};

}  // namespace vptq::kernels::copy
