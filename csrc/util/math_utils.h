// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq {

HOST bool is_pow2(unsigned int x) { return ((x & (x - 1)) == 0); }

template <typename T, typename X, typename Y>
HOST T divup(const X x, const Y y) {
  return static_cast<T>((x + y - 1) / y);
}

HOST unsigned int next_pow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

HOST unsigned int log2_floor(unsigned int x) {
  if (x == 0) return -1U;
  int log = 0;
  unsigned int value = x;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    unsigned int n = value >> shift;
    if (n != 0) {
      value = n;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

}  // namespace vptq
