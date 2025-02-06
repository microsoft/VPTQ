// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.cuh"

#include <cuda_runtime_api.h>

namespace vptq {

DEVICE bool block(int bid) {
  int id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  return id == bid;
}

DEVICE bool thread(int tid, int bid) {
  int id = threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
  return id == tid && block(bid);
}

// usage, e.g.
// if (thread(0, 0)) { ... }
// if (thread(37)) { ... }
// if (block(0)) { ... }

DEVICE bool thread(int tid) { return thread(tid, 0); }

DEVICE bool thread0() { return thread(0, 0); }

DEVICE bool block0() { return block(0); }

}  // namespace vptq
