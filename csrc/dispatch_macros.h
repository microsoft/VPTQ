// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define DISPATCH_TYPE_CASE(TYPE, NV_TYPE, ...)                               \
  case TYPE: {                                                               \
    std::cout << "Dispatching for type(2): " << toString(TYPE) << std::endl; \
    using nv_type = NV_TYPE;                                                 \
    return __VA_ARGS__();                                                    \
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
