// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels {

#define FULL_WARP_MASK 0xFFFFFFFF

#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))

template <typename T, typename Reducer>
DEVICE T wrap_reduce(T val, unsigned mask, Reducer reducer) {
  val = reducer(val, __shfl_down_sync(mask, val, 16, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 8, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 4, 32));
  val = reducer(val, __shfl_down_sync(mask, val, 2, 32));
  return reducer(val, __shfl_down_sync(mask, val, 1, 32));
}

// NOTE: This function works only for arrays with sizes that are powers of 2.
template <typename T, typename Reducer>
DEVICE T power2_reduce(T val, int tid, T* shm, Reducer reducer, int block_size,
                       T init_val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < block_size);
  val = wrap_reduce(val, mask, reducer);

  if (tid < WARP_SIZE) shm[tid] = init_val;
  __syncthreads();

  if (tid % WARP_SIZE == 0) shm[tid / WARP_SIZE] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < WARP_SIZE);
  if (tid < WARP_SIZE) {
    val = shm[tid];
    val = wrap_reduce(val, mask, reducer);
  }
  return val;
}

}  // namespace vptq::kernels
