// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace vptq {

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define gpuErrchk(ret) gpuAssert((ret), __FILE__, __LINE__);

template <typename T>
__forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

class OptionalCUDAGuard {
  int set_device_ = -1;
  int current_device_ = -1;

public:
  OptionalCUDAGuard(int device) : set_device_(device) {
    cudaError_t err = cudaGetDevice(&current_device_);
    std::stringstream ss;
    if (err != cudaSuccess) {
      ss << "cudaGetDevice failed with error code " << cudaGetErrorString(err);
      TORCH_CHECK(err == cudaSuccess, ss.str());
    }
    if (current_device_ == device) {
      return;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      ss << "cudaGetDevice failed with error code " << cudaGetErrorString(err);
      TORCH_CHECK(err == cudaSuccess, ss.str());
    }
  }
  ~OptionalCUDAGuard() {
    if (set_device_ != current_device_) cudaSetDevice(current_device_);
  }
};

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    TORCH_CHECK(false, cudaGetErrorString(code));
  }
}

}  // namespace vptq
