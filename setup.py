# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

from packaging.version import Version, parse
from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from torch.utils.cpp_extension import CUDA_HOME

cur_path = Path(__file__).parent


def get_version():
    with open(cur_path / "pyproject.toml") as f:
        for line in f:
            if "version" in line:
                return line.split("=")[-1].strip().strip('"')
    return "0.0.1"


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def nvcc_threads():
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        nvcc_threads = os.getenv("NVCC_THREADS") or (os.cpu_count() // 2)
        return nvcc_threads


class CMakeExtension(Extension):
    """specify the root folder of the CMake projects"""

    def __init__(self, name, cmake_lists_dir=".", **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

        if os.path.isdir(".git"):
            subprocess.run(
                ["git", "submodule", "update", "--init", "third_party/cutlass"],
                check=True,
            )
        else:
            if not os.path.exists(
                "third_party/cutlass/include/cutlass/cutlass.h"
            ):
                raise RuntimeError(
                    (
                        "third_party/cutlass is missing, "
                        "please use source distribution or git clone"
                    )
                )


class CMakeBuildExt(build_ext):
    """launches the CMake build."""

    def copy_extensions_to_source(self) -> None:
        pass

    def build_extension(self, ext: CMakeExtension) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable") from None

        debug = (
            int(os.environ.get("DEBUG", 0))
            if self.debug is None
            else self.debug
        )
        cfg = "Debug" if debug else "Release"

        # Set CUDA_ARCH_LIST to build the shared library
        # for the specified GPU architectures.
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", None)
        if parallel_level is not None:
            self.parallel = int(parallel_level)
        else:
            self.parallel = os.cpu_count()

        for ext in self.extensions:
            # Get the package directory where the library should be installed
            package_dir = os.path.join(self.build_lib, "vptq")
            os.makedirs(package_dir, exist_ok=True)

            # Create build directory for this extension
            build_temp = Path(self.build_temp) / ext.name
            if not build_temp.exists():
                build_temp.mkdir(parents=True)

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(str(package_dir)),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(str(build_temp)),
                (
                    "-DUSER_CUDA_ARCH_LIST={}".format(arch_list)
                    if arch_list
                    else ""
                ),
                "-DNVCC_THREADS={}".format(nvcc_threads()),
            ]

            # Adding CMake arguments set as environment variable
            if "CMAKE_ARGS" in os.environ:
                cmake_args += [
                    item for item in os.environ["CMAKE_ARGS"].split(" ") if item
                ]

            build_args = []
            build_args += ["--config", cfg]
            # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
            # across all generators.
            if (
                "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ
                and hasattr(self, "parallel")
                and self.parallel
            ):
                build_args += [f"-j{self.parallel}"]

            # Config
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=str(build_temp)
            )

            # Build
            subprocess.check_call(
                ["cmake", "--build", "."] + build_args, cwd=str(build_temp)
            )

            # Verify the library was built
            target = "libvptq.so"
            target_path = Path(package_dir) / target

            if not target_path.exists():
                raise FileNotFoundError(
                    f"Library was not built in the expected location: {target_path}"
                )


class Develop(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)


class Clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # clean the dynamic library if it exists in the source directory
        # the dynamic library might be copied to the source directory
        # under the develop mode
        lib_path = Path("vptq") / "libvptq.so"
        if lib_path.exists():
            print(f"cleaning dynamic library '{lib_path}'")
            try:
                os.remove(lib_path)
            except OSError as e:
                print(f"Warning: Could not remove {lib_path}: {e}")

        # Then clean other files based on .gitignore
        with open(".gitignore") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip("./")

                    for filename in glob.glob(wildcard):
                        print(f"cleaning '{filename}'")
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


description = (
    "VPTQ: Extreme Low-bit Vector Post-Training Quantization "
    "for Large Language Models"
)

setup(
    name="vptq",
    python_requires=">=3.8",
    packages=find_packages(exclude=[""]),
    version=get_version(),
    description=description,
    author="Wang Yang, Wen JiCheng, Cao Ying",
    ext_modules=[CMakeExtension("vptq")],
    cmdclass={
        "build_ext": CMakeBuildExt,
        "clean": Clean,
        "develop": Develop,
    },
)
