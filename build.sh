#!/bin/bash

# build_type="debug"
build_type="release"
export COMPILER_PATH=/mnt/development/toolkit/compilers/gcc-arm-11.2-2022.02-x86_64-arm-none-linux-gnueabihf/bin/

DIR=./
if [ ! -d "./${DIR}/build" ]; then
    mkdir ./${DIR}/build
fi

cd ./${DIR}/build

cmake -DCMAKE_TOOLCHAIN_FILE=cmake_includes/toolchain.cmake -DCMAKE_BUILD_TYPE=$build_type ../

#make
cmake --build . -- -j 10