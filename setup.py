# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
from pathlib import Path

from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

cur_path = Path(__file__).parent


def get_version():
    with open(cur_path / "pyproject.toml") as f:
        for line in f:
            if "version" in line:
                return line.split("=")[-1].strip().strip('"')
    return "0.0.1"


def get_requirements():
    """Get Python package dependencies from requirements.txt."""
    with open(cur_path / "requirements.txt") as f:
        requirements = f.read().strip().split("\n")
    requirements = [req for req in requirements if "https" not in req]
    return requirements


class CMakeExtension(Extension):
    """ specify the root folder of the CMake projects"""

    def __init__(self, name, cmake_lists_dir=".", **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


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

        debug = int(
            os.environ.get("DEBUG", 0)
        ) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", None)
        if parallel_level is not None:
            self.parallel = int(parallel_level)
        else:
            self.parallel = os.cpu_count()

        for ext in self.extensions:
            extdir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name))
            )
            extdir = os.path.join(extdir, "vptq")

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), extdir
                ), "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                )
            ]

            # Adding CMake arguments set as environment variable
            if "CMAKE_ARGS" in os.environ:
                cmake_args += [
                    item for item in os.environ["CMAKE_ARGS"].split(" ") if item
                ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            build_args = []
            build_args += ["--config", cfg]
            # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
            # across all generators.
            if (
                "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ and
                hasattr(self, "parallel") and self.parallel
            ):
                build_args += [f"-j{self.parallel}"]

            build_temp = Path(self.build_temp) / ext.name
            if not build_temp.exists():
                build_temp.mkdir(parents=True)

            # Config
            subprocess.check_call(["cmake", ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            # Build
            subprocess.check_call(["cmake", "--build", "."] + build_args,
                                  cwd=self.build_temp)


class Develop(develop):
    """Post-installation for development mode."""

    def post_build_copy(self) -> None:
        build_py = self.get_finalized_command("build_py")
        source_root = Path(os.path.abspath(build_py.build_lib)).parents[1]

        target = "libvptq.so"

        source_path = os.path.join(build_py.build_lib, "vptq", target)
        target_path = os.path.join(source_root, "vptq", target)

        if os.path.exists(source_path):
            self.copy_file(source_path, target_path, level=self.verbose)
        else:
            raise FileNotFoundError(f"Cannot find built library: {source_path}")

    def run(self):
        develop.run(self)
        self.post_build_copy()


class Clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re
        import shutil

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
    install_requires=get_requirements(),
    version=get_version(),
    description=description,
    author="Wang Yang, Wen JiCheng",
    ext_modules=[CMakeExtension("vptq")],
    cmdclass={
        "build_ext": CMakeBuildExt,
        "clean": Clean,
        "develop": Develop,
    },
)
