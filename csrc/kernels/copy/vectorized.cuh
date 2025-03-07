// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels::copy {

namespace {
struct __align__(8) __half4 {
  __half2 x;
  __half2 y;
};

struct __align__(16) __half8 {
  __half2 x;
  __half2 y;
  __half2 z;
  __half2 w;
};

struct __align__(8) __bfloat4 {
  __bfloat162 x;
  __bfloat162 y;
};

struct __align__(16) __bfloat8 {
  __bfloat162 x;
  __bfloat162 y;
  __bfloat162 z;
  __bfloat162 w;
};
}  // namespace

template <typename DType, int kN>
struct GetPackType;

template <>
struct GetPackType<__half, 2> {
  using type = __half2;
};

template <>
struct GetPackType<__half, 4> {
  using type = __half4;
};

template <>
struct GetPackType<__half, 8> {
  using type = __half8;
};

template <>
struct GetPackType<__bfloat16, 2> {
  using type = __bfloat162;
};

template <>
struct GetPackType<__bfloat16, 4> {
  using type = uint2;  // TODO(ying): verify the generated ptx
};

template <>
struct GetPackType<__bfloat16, 8> {
  using type = uint4;  // TODO(ying): verify the generated ptx
};

template <>
struct GetPackType<uint8_t, 4> {
  using type = int;
};

template <>
struct GetPackType<uint16_t, 2> {
  using type = int;
};

template <>
struct GetPackType<uint16_t, 4> {
  using type = int2;
};

template <>
struct GetPackType<uint, 4> {
  using type = uint4;  // uint4 has native 128 bits load/store support
};

template <>
struct GetPackType<float4, 4> {
  using type = float4;  // float4 has native 128 bits load/store support
};

template <typename DType, int kN>
using PackType = typename GetPackType<DType, kN>::type;

/// Vectorized copy for a single access.
/// @param DType_ The data type of the elements to copy.
/// @param kN The number of elements to pack into a vectorized copy. This
///        should be no more than 128 bits.
template <typename DType_, int kN>
struct PackedCopy {
  using DType = DType_;
  using Packed = PackType<DType, kN>;

  // the maximum read/write transaction size in bytes for a thread
  static constexpr int kMaxVecBytes = 16;

  static_assert(sizeof(DType) * kN <= kMaxVecBytes,
                "The total number of bytes must be less than or equal to the "
                "maximum width of a vectorized instruction.");

  // This ctor does nothing but ensures the object is created in device memory
  DEVICE PackedCopy() {}

  DEVICE void operator()(const DType* src_, DType* dst_) {
    const Packed* src = reinterpret_cast<const Packed*>(src_);
    Packed* dst = reinterpret_cast<Packed*>(dst_);
    *dst = *src;
  }
};

}  // namespace vptq::kernels::copy
