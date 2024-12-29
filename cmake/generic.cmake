# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard whose features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    17
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python interpreter path: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")

include_directories(${Python3_INCLUDE_DIRS})

set(TORCH_LIB_PREFIX "${Python3_SITEARCH}/torch")
if(NOT EXISTS ${TORCH_LIB_PREFIX})
  message(FATAL_ERROR "Torch library is not installed.")
else()
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB_PREFIX}/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)

message(STATUS "Torch include include_directories: " ${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

# let cmake automatically detect the current CUDA architecture to avoid
# generating device codes for all possible architectures
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --Werror all-warnings")
# Set the CUDA_PROPAGATE_HOST_FLAGS to OFF to avoid passing host compiler flags
# to the device compiler
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# FIXME(ying): -std=c++17 has to be set explicitly here, Otherwise, linking
# against torchlibs will raise errors. it seems that the host compilation
# options are not passed to torchlibs.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -std=c++17 -O0)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -std=c++17 -O3)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__
    -U__CUDA_NO_BFLOAT162_CONVERSIONS__)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} "
          "is greater or equal than 11.0, enable -DENABLE_BF16 flag.")
endif()

message(STATUS "CUDA detected: " ${CUDA_VERSION})
message(STATUS "CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
