// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// register VPTQ APIs bindings in this file. ///

#include "dequant.h"
#include "quant_gemv.h"

// NOTE: DO NOT change the module name "libvptq" here. It must match how
// the module is loaded in the Python codes.
PYBIND11_MODULE(libvptq, m) {
  m.doc() = "VPTQ customized kernels.";

  // v1 kernels.
  m.def("dequant", &vptq::dequant, "vptq customized dequantization kernel.");
  m.def("quant_gemv", &vptq::wquant_act16_gemv,
        "vptq customized dequantized gemv kernel.");

  // v2 kernels.
  m.def("quant_gemv_v2", &vptq::quant_gemv_v2,
        "vptq customized quantized gemm kernel.");
}
