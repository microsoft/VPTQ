// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config.cuh"

namespace vptq::kernels {

template <typename T>
DEVICE T from_float(float v, T vv) {  // vv is for type deduction
  (void)(vv);
  if constexpr (std::is_same<T, __bfloat16>::value) {
    return vv = __float2bfloat16(v);
  } else if constexpr (std::is_same<T, __half>::value) {
    return vv = __float2half(v);
  } else {
    static_assert(std::is_same<T, float>::value);
    return vv = v;
  }
}

template <typename T>
DEVICE float to_float(T v) {
  return static_cast<float>(v);
}

template <>
DEVICE float to_float(__bfloat16 v) {
  return __bfloat162float(v);
}

template <>
DEVICE float to_float(__half v) {
  return __half2float(v);
}

}  // namespace vptq::kernels
