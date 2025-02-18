// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace vptq::kernels::copy {

template <int N>
DEVICE void wait_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

DEVICE void commit_copy_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

DEVICE void __copy_async() {
  commit_copy_group();
  wait_group<0>();
}

}  // namespace vptq::kernels::copy
