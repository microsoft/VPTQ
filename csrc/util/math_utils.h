// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq {
template <typename T, typename X, typename Y>
HOST T divup(const X x, const Y y) {
  return static_cast<T>((x + y - 1) / y);
}
}  // namespace vptq
