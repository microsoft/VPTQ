// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/copy/layout.cuh"

namespace vptq::kernels::copy {

template <const int kNumWarps, const int kNumWarpsPerTile>
struct WarpCounter {
  HOST_DEVICE WarpCounter() : cur_warp_(0), next_warp_(kNumWarpsPerTile) {}

  HOST_DEVICE void reset() {
    cur_warp_ = 0;
    next_warp_ = kNumWarpsPerTile;
  }

  HOST_DEVICE int cur() const { return cur_warp_; }

  HOST_DEVICE int next() const { return next_warp_; }

  HOST_DEVICE void operator++() {  // TODO(ying): simplify these calculations
    cur_warp_ = next_warp_ % kNumWarps;
    next_warp_ += kNumWarpsPerTile;
    next_warp_ = next_warp_ > kNumWarps ? next_warp_ % kNumWarps : next_warp_;
  }

private:
  int cur_warp_;
  int next_warp_;
};

}  // namespace vptq::kernels::copy
