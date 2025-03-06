#!/bin/bash

# rm -rf dist
# python3 setup.py build bdist_wheel

# python3 setup.py clean

# export TORCH_CUDA_ARCH_LIST="7.5"
# export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
# python3 setup.py build bdist_wheel 2>&1 | tee build.log

# unset TORCH_CUDA_ARCH_LIST
# export NVCC_THREADS=`nproc`

# python3 setup.py clean
# python3 setup.py develop 2>&1 | tee build.log
# exit 0

# TORCH_VERSION=`python -c "import torch; print(torch.__version__)"`
# python3 setup.py build bdist_wheel \
#   --dist-dir="../torch$TORCH_VERSION-dist" 2>&1 | tee build.log

root_dir=`pwd`

build_dir="$root_dir/build/temp.linux-x86_64-cpython-312"
if [ ! -d "$build_dir" ]; then
   mkdir -p $build_dir
fi
cd $build_dir

lib_dir="$build_dir/csrc"
# lib_dir="$root_dir/build/lib.linux-x86_64-cpython-312/vptq"
source_file="$lib_dir/libvptq.so"

if [ -f "$source_file" ]; then
   echo "[1]: deleted old libvptq.so"
   rm "$source_file"
fi

if [ -d "CMakeFiles" ]; then
  rm -rf CMakeFiles
fi

if [ -f "CMakeCache.txt" ]; then
  rm -f CMakeCache.txt
fi

cmake -DCMAKE_C_COMPILER=`which gcc` \
   -DCMAKE_CXX_COMPILER=`which g++` \
   ../../

make -j32 2>&1 | tee ../../build.log

if [ ! -f "$source_file" ]; then
   echo "build failed"
   exit 1
else
   echo "build success"
fi

target_file="$root_dir/vptq/libvptq.so"
if [ -f "$target_file" ]; then
   rm "$target_file"
fi

cp $source_file $target_file
cd ../../

if [ -f "$target_file" ]; then
   echo "running test..."
   python3 vptq/tests/ops/test_quant_gemv.py 2>&1 | tee test.log
else
   echo "libvptq.so not found" 
fi
