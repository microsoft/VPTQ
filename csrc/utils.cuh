// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>


namespace cuda {

constexpr int kBlockSize = 256;

template <typename T> struct TypeVec2 {
  typedef __half2 type;
};

template<> struct TypeVec2<__nv_bfloat16> {
  typedef __nv_bfloat162 type;
};

template <typename T>
T __device__ __forceinline__ ConvertFromFloat(float v){
  if constexpr (std::is_same_v<T, __nv_bfloat16>){
    return __float2bfloat16(v);
  }
  return __float2half(v);
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if constexpr (WarpSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if constexpr (WarpSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if constexpr (WarpSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if constexpr (WarpSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if constexpr (WarpSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

template<int GROUPSIZE>
__device__ __forceinline__ void ldg_vec_x(uint32_t* __restrict__ dst_u32 , const uint32_t* __restrict__ src_u32){
  int2* dst = (int2*)dst_u32;
  const int2* src = (const int2*)src_u32;
  if constexpr (GROUPSIZE == 2){
   *dst_u32 = __ldg(src_u32);
    //uint32_t* dec = (uint32_t*)dst;
    //asm volatile (
    //      "ld.cg.global.v2.u32 {%0, %1}, [%2];"
    //      : "=r"(dec[0]), "=r"(dec[1])
    //      : "l"((const void*)src)
    //    );
   } else if constexpr (GROUPSIZE == 4){
    *dst = __ldg(src);
    //uint32_t* dec = (uint32_t*)dst;
    //asm volatile (
    //      "ld.cg.global.v2.u32 {%0, %1}, [%2];"
    //      : "=r"(dec[0]), "=r"(dec[1])
    //      : "l"((const void*)src)
    //    );
  }else if constexpr (GROUPSIZE == 6){
    dst_u32[0] = __ldg(src_u32);
    dst_u32[1] = __ldg(src_u32+1);
    dst_u32[2] = __ldg(src_u32+2);
  } else if constexpr (GROUPSIZE == 8){
      *(int4*)dst = __ldg((const int4*)src);
  } else if constexpr (GROUPSIZE == 16){
      // *(int4*)dst = __ldg((const int4*)src);
      // *(int4*)(dst+2) = __ldg((const int4*)(src+2));
      asm volatile (
             "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
             : "=r"(dst_u32[0]), "=r"(dst_u32[1]), "=r"(dst_u32[2]), "=r"(dst_u32[3])
             : "l"((const void*)src_u32)
           );
      asm volatile (
           "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
           : "=r"(dst_u32[4]), "=r"(dst_u32[5]), "=r"(dst_u32[6]), "=r"(dst_u32[7])
           : "l"((const void*)(src_u32+4))
         );
    }else if constexpr (GROUPSIZE == 12){      
      if (uint64_t(src)%16){
        dst[0] = __ldg(src);
        int4 b = __ldg((int4*)(src+1));
        dst[1] = *((int2*)&b);
        dst[2] = *((int2*)&b+1);
      }else{
        *(int4*)dst = __ldg((int4*)(src));
        dst[2] = __ldg((src+2));
      }
      // dst[0] = __ldg(src);
      // dst[1] = __ldg((src+1));
      // dst[2] = __ldg((src+2));

      // uint32_t* dec = (uint32_t*)dst;
      // asm volatile (
      //         "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      //         : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
      //         : "l"((const void*)src)
      //       );
      // asm volatile (
      //       "ld.cg.global.v2.u32 {%0, %1}, [%2];"
      //       : "=r"(dec[4]), "=r"(dec[5])
      //       : "l"((const void*)src)
      //     );
    }
}

template <int WBITS>
__device__ __forceinline__ uint32_t iterator_packed_tensor(const uint32_t* ptr, int idx) {
  if constexpr (WBITS == 32) {
    return ptr[idx];
  } else if constexpr (WBITS == 16) {
    return ((const uint16_t*)ptr)[idx];
  } else {
    int start_bits = idx * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;
    uint32_t sec_v = ptr[second];
    uint32_t v = (ptr[first] >> (start_bits)) & ((1 << WBITS) - 1);
    if (first == second) {
      return v;
    } else {
      v |= ((sec_v) & ((1 << (end_bits)) - 1))<< (32-start_bits);
      return v;
    }
  }
}

template <typename T> __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace cuda
