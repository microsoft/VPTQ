# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vptq LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(generic)

# set the third party directory for dependencies that do not need a build
set(THIRD_PARTY_DIR
    "${PROJECT_SOURCE_DIR}/third_party"
    CACHE STRING
          "A path that specifies the directory for third-party downloads.")
# add cutlass into dependence
include_directories(${THIRD_PARTY_DIR}/cutlass/include)
include_directories(csrc)

add_subdirectory(csrc)
