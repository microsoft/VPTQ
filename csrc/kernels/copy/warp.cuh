// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels::copy {

template <const int kNumWarps, const int kStep = 1>
struct WarpCounter {
  HOST_DEVICE WarpCounter() : cur_warp_(0), next_warp_(kStep) {}

  HOST_DEVICE void reset() {
    cur_warp_ = 0;
    next_warp_ = kStep;
  }

  // TODO(ying): simplify these calculations
  HOST_DEVICE int cur() const { return cur_warp_; }

  HOST_DEVICE int next() const { return next_warp_; }

  HOST_DEVICE void operator++() {
    cur_warp_ = next_warp_ % kNumWarps;
    next_warp_ += kStep;
    next_warp_ = next_warp_ > kNumWarps ? next_warp_ % kNumWarps : next_warp_;
  }

  HOST_DEVICE WarpCounter& operator+=(int n) {
    cur_warp_ = next_warp_ % kNumWarps;
    next_warp_ += n;
    next_warp_ = next_warp_ > kNumWarps ? next_warp_ % kNumWarps : next_warp_;
    return *this;
  }

private:
  int cur_warp_;
  int next_warp_;
};

}  // namespace vptq::kernels::copy
