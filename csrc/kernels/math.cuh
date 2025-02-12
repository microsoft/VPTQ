// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels {

template <typename T>
struct Sum {
  HOST_DEVICE T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct Max {
  HOST_DEVICE T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

template <typename T>
struct Min {
  HOST_DEVICE T operator()(const T& a, const T& b) const {
    return a > b ? b : a;
  }
};

template <typename T>
struct Prod {
  HOST_DEVICE T operator()(const T& a, const T& b) const { return a * b; }
};

}  // namespace vptq::kernels
