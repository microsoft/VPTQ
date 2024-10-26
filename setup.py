# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cur_path = Path(__file__).parent


def get_version():
    with open(cur_path / "vptq/__init__.py") as f:
        for line in f:
            if "__version__" in line:
                return line.split("=")[-1].strip().strip('"')
    return "0.0.1"


def build_cuda_extensions():
    if os.getenv("SKIP_COMPILE", "0") == "1":
        print("Skipping CUDA/ROCm extension compilation due to SKIP_COMPILE setting")
        return []
    
    compute_capabilities = [70, 75, 80, 86, 89, 90]
    
    arch_flags = []
    TORCH_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", None)
    if TORCH_CUDA_ARCH_LIST is None:
        print("TORCH_CUDA_ARCH_LIST is not set, compiling for all arch")
    else:
        delimiter = ' ' if ';' not in TORCH_CUDA_ARCH_LIST else ' '
        TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST.split(delimiter)
        compute_capabilities = [int(10 * float(arch)) for arch in TORCH_CUDA_ARCH_LIST if '+' not in arch]

    if torch.cuda.is_available() and torch.version.hip is not None:
        PYTORCH_ROCM_ARCH = os.getenv("PYTORCH_ROCM_ARCH", None)
        arch_name = torch.cuda.get_device_properties().gcnArchName.split(":")[0]
        if PYTORCH_ROCM_ARCH is not None and arch_name not in PYTORCH_ROCM_ARCH:
            PYTORCH_ROCM_ARCH = PYTORCH_ROCM_ARCH + f";{arch_name}"
        elif PYTORCH_ROCM_ARCH is None:
            PYTORCH_ROCM_ARCH = arch_name
        compute_capabilities = PYTORCH_ROCM_ARCH
        os.environ["PYTORCH_ROCM_ARCH"] = PYTORCH_ROCM_ARCH
    else:
        for cap in compute_capabilities:
            arch_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]
    print(" build for compute capabilities: ==============", compute_capabilities)

    extra_compile_args = {
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-DENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        ] + arch_flags,
        "cxx": ["-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    }

    if torch.cuda.is_available() and torch.version.hip is not None:
        extra_compile_args["nvcc"].extend(["-fbracket-depth=1024"])
    else:
        extra_compile_args["nvcc"].extend(
            ["--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-lineinfo"])

    extensions = CUDAExtension(
        "vptq.ops",
        [
            "csrc/ops.cc",
            "csrc/dequant_impl_packed.cu",
        ],
        extra_compile_args=extra_compile_args,
    )
    return [extensions]


def get_requirements():
    """Get Python package dependencies from requirements.txt."""
    with open(cur_path / "requirements.txt") as f:
        requirements = f.read().strip().split("\n")
    requirements = [req for req in requirements if "https" not in req]
    return requirements


setup(
    name="vptq",
    python_requires=">=3.8",
    packages=find_packages(exclude=[""]),
    install_requires=get_requirements(),
    version=get_version(),
    ext_modules=build_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
