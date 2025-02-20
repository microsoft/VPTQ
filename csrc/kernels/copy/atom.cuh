// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdint.h>

namespace vptq::kernels::copy {

namespace {
/// ld.shared
template <const int kBytes>
DEVICE void ld_shared(void* dst, uint32_t src);

/// ld.shared - 16b
template <>
DEVICE void ld_shared<2>(void* dst, uint32_t src) {
  asm volatile("ld.shared.u16 %0, [%1];\n"
               : "=h"(*reinterpret_cast<uint16_t*>(dst))
               : "r"(src));
}

/// ld.shared - 32b
template <>
DEVICE void ld_shared<4>(void* dst, uint32_t src) {
  asm volatile("ld.shared.u32 %0, [%1];\n"
               : "=r"(*reinterpret_cast<uint32_t*>(dst))
               : "r"(src));
}

/// ld.shared - 64b
template <>
DEVICE void ld_shared<8>(void* dst, uint32_t src) {
  uint2* dst_u64 = reinterpret_cast<uint2*>(dst);
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
               : "=r"(dst_u64->x), "=r"(dst_u64->y)
               : "r"(src));
}

/// ld.shared - 128b
template <>
DEVICE void ld_shared<16>(void* dst, uint32_t src) {
  uint4* dst_u128 = reinterpret_cast<uint4*>(dst);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z),
                 "=r"(dst_u128->w)
               : "r"(src));
}

/// st.shared
template <int kBytes>
DEVICE void st_shared(uint32_t dst, void const* src);

/// st.shared - 16b
template <>
DEVICE void st_shared<2>(uint32_t dst, void const* src) {
  asm volatile("st.shared.u16 [%0], %1;\n"
               :
               : "r"(dst), "h"(*reinterpret_cast<uint16_t const*>(src)));
}

/// st.shared - 32b
template <>
DEVICE void st_shared<4>(uint32_t dst, void const* src) {
  asm volatile("st.shared.u32 [%0], %1;\n"
               :
               : "r"(dst), "r"(*reinterpret_cast<uint32_t const*>(src)));
}

/// st.shared - 64b
template <>
DEVICE void st_shared<8>(uint32_t dst, void const* src) {
  uint2 const* dst_u64 = reinterpret_cast<uint2 const*>(src);
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
               :
               : "r"(dst), "r"(dst_u64->x), "r"(dst_u64->y));
}

/// st.shared - 128b
template <>
DEVICE void st_shared<16>(uint32_t dst, void const* src) {
  uint4 const* dst_u128 = reinterpret_cast<uint4 const*>(src);
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(dst), "r"(dst_u128->x), "r"(dst_u128->y), "r"(dst_u128->z),
                 "r"(dst_u128->w));
}

/// st.global
template <int kBytes>
DEVICE void st_global(void* dst, const void* src);

template <>
DEVICE void st_global<16>(void* dst, const void* src) {
  uint4 const* dst_u128 = reinterpret_cast<uint4 const*>(src);
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(dst), "r"(dst_u128->x), "r"(dst_u128->y), "r"(dst_u128->z),
                 "r"(dst_u128->w));
}
}  // namespace

template <int kBytes>
DEVICE void ld_shared_st_global(void* dst, uint32_t src);

template <>
DEVICE void ld_shared_st_global<16>(void* dst, uint32_t src) {
  unsigned tmp[4];
  ld_shared<16>(tmp, src);
  st_global<16>(dst, tmp);
}

template <const int kBytes>
DEVICE void ld_global_st_shared(uint32_t dst, void const* src) {
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16);

#if (__CUDA_ARCH__ >= 800)
  // SM90, hopper, SM80, SM86, ampere

  // TODO(ying): add a wrapper to allow choosing between different caching
  // policies (e.g. "cache all levels").
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst),
               "l"(src), "n"(kBytes));
#else
  unsigned tmp[kBytes / 4];
  if constexpr (kBytes == 16) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                 : "l"(src));
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst),
                 "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]));
  } else if constexpr (kBytes == 8) {
    asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n"
                 : "=r"(tmp[0]), "=r"(tmp[1])
                 : "l"(src));
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(dst), "r"(tmp[0]),
                 "r"(tmp[1]));
  } else if constexpr (kBytes == 4) {
    asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(tmp[0]) : "l"(src));
    asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(dst), "r"(tmp[0]));
  }
#endif
}

}  // namespace vptq::kernels::copy
