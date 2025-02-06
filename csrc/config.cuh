// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if defined(__CUDA_ARCH__)
  #define HOST_DEVICE __forceinline__ __host__ __device__
  #define DEVICE __forceinline__ __device__
  #define HOST __forceinline__ __host__
#else
  #define HOST_DEVICE inline
  #define DEVICE inline
  #define HOST inline
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  #define CP_ASYNC_SM80_ENABLED
#endif

#if defined(USE_ROCM)
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  #define VPTQ_LDG(arg) __ldg(arg)
  #define SHFL_DOWN(val, offset) __shfl_down(val, offset)
  #define WARP_SIZE warpSize

typedef __hip_bfloat162 __bfloat162;
typedef __hip_bfloat16 __bfloat16;

#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>

  #define WARP_SIZE 32
  #define VPTQ_LDG(arg) *(arg)
  #define SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)

typedef __nv_bfloat162 __bfloat162;
typedef __nv_bfloat16 __bfloat16;

#endif
