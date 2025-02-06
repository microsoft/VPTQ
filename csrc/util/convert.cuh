// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config.cuh"

namespace vptq {
template <typename T>
T DEVICE from_float(float v, T vv) {
  (void)(vv);
  if constexpr (std::is_same<T, __bfloat16>::value) {
    return vv = __float2bfloat16(v);
  } else if constexpr (std::is_same<T, float>::value) {
    return vv = v;
  } else {
    static_assert(std::is_same<T, __half>::value);
    return vv = __float2half(v);
  }
}

template <typename T>
float DEVICE to_float(T v) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
    return __bfloat162float(v);
  } else if constexpr (std::is_same<T, float>::value) {
    return v;
  } else {
    static_assert(std::is_same<T, __half>::value);
    return __half2float(v);
  }
}
}  // namespace vptq
