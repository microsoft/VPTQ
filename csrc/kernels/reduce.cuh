// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "kernels/convert.cuh"

namespace vptq::kernels {

template <typename DType>
struct Sum {
  HOST_DEVICE DType operator()(const DType& a, const DType& b) const {
    return a + b;
  }
};

#define FULL_WARP_MASK 0xFFFFFFFF

#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))

template <typename T, typename Reducer>
DEVICE T wrap_reduce(T val, unsigned mask, Reducer reducer) {
#if defined(__CUDA_ARCH__)
  val = reducer(val, __shfl_down_sync(mask, val, 16, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 8, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 4, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 2, 32));
  return reducer(val, __shfl_down_sync(mask, val, 1, 32));
#else
  return val;
#endif
}

// NOTE: This function works only for arrays with sizes that are powers of 2.
// NOTE: `shm` must be initialized to `init_val` before calling this function.
template <typename T, typename Reducer>
DEVICE T power2_reduce(T val, T* shm, Reducer reducer, T init_val) {
#if defined(__CUDA_ARCH__)
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < block_size);
  val = wrap_reduce(val, mask, reducer);

  // TODO: multiple threads access a single bank will cause bank conflict
  if (tid % WARP_SIZE == 0) shm[tid / WARP_SIZE] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < WARP_SIZE);
  if (tid < WARP_SIZE) {
    val = shm[tid];
    val = wrap_reduce(val, mask, reducer);
  }
  return val;
#else
  assert(false && "power2_reduce is not supported on CPU");
  return static_cast<T>(0);
#endif
}

}  // namespace vptq::kernels
