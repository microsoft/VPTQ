// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define DISPATCH_TYPE_CASE(TYPE, NV_TYPE, ...) \
  case TYPE: {                                 \
    using DType = NV_TYPE;                     \
    return __VA_ARGS__();                      \
  }

#define VPTQ_DISPATCH_TYPES(TYPE, ...)                                       \
  c10::ScalarType _type = TYPE;                                              \
  [&] {                                                                      \
    switch (_type) {                                                         \
      DISPATCH_TYPE_CASE(c10::ScalarType::Half, __half, __VA_ARGS__)         \
      DISPATCH_TYPE_CASE(c10::ScalarType::BFloat16, __bfloat16, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR("Dispatch is not implemented for type: '", toString(_type), \
                 "'");                                                       \
    }                                                                        \
  }();

// TODO(ying): Add support for kVecLen = 12
#define VPTQ_DISPATCH_VEC_LENGTH(VEC_LEN, ...)                                \
  [&] {                                                                       \
    switch (VEC_LEN) {                                                        \
      case 4: {                                                               \
        static constexpr int kVecLen = 4;                                     \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 8: {                                                               \
        static constexpr int kVecLen = 8;                                     \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 16: {                                                              \
        static constexpr int kVecLen = 16;                                    \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        AT_ERROR("Dispatch is not implemented for vector length: ", VEC_LEN); \
    }                                                                         \
  }();

#define VPTQ_DISPATCH_NUM_CENTROIDS(NUM_CENTROIDS, ...)                \
  [&] {                                                                \
    switch (NUM_CENTROIDS) {                                           \
      case 4096: {                                                     \
        static constexpr int kNumCentroids = 4096;                     \
        using IdType = uint16_t;                                       \
        return __VA_ARGS__();                                          \
      }                                                                \
      case 8192: {                                                     \
        static constexpr int kNumCentroids = 8192;                     \
        using IdType = uint16_t;                                       \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        AT_ERROR("Dispatch is not implemented for centroids number: ", \
                 NUM_CENTROIDS);                                       \
    }                                                                  \
  }();

#define VPTQ_DISPATCH_RES_NUM_CENTROIDS(NUM_RES_CENTROIDS, ...)        \
  [&] {                                                                \
    switch (NUM_RES_CENTROIDS) {                                       \
      case 0: {                                                        \
        static constexpr int kNumResCentroids = 0;                     \
        using ResIdType = uint8_t;                                     \
        return __VA_ARGS__();                                          \
      }                                                                \
      case 256: {                                                      \
        static constexpr int kNumResCentroids = 256;                   \
        using ResIdType = uint8_t;                                     \
        return __VA_ARGS__();                                          \
      }                                                                \
      case 512: {                                                      \
        static constexpr int kNumResCentroids = 512;                   \
        using ResIdType = uint16_t;                                    \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        AT_ERROR("Dispatch is not implemented for centroids number: ", \
                 NUM_RES_CENTROIDS);                                   \
    }                                                                  \
  }();
