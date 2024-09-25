
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

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

#define gpuErrchk(ret) gpuAssert((ret), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    TORCH_CHECK(false, cudaGetErrorString(code));
  }
}