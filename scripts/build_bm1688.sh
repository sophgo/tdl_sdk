#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVI_TDL_ROOT=$(readlink -f $SCRIPT_DIR/../)
SDK_VER=64bit
TMP_WORKING_DIR=$CVI_TDL_ROOT/tmp
BUILD_WORKING_DIR=$TMP_WORKING_DIR/build_sdk
BUILD_DOWNLOAD_DIR=$TMP_WORKING_DIR/_deps
AI_SDK_INSTALL_PATH=$CVI_TDL_ROOT/install
CHIP_ARCH=BM1688
USE_TPU_IVE=OFF

if exists $AI_SDK_INSTALL_PATH; then
    echo "AI_SDK_INSTALL_PATH=$AI_SDK_INSTALL_PATH"
else
    mkdir -p $AI_SDK_INSTALL_PATH
fi

if exists $BUILD_WORKING_DIR; then
    echo "BUILD_WORKING_DIR=$BUILD_WORKING_DIR"
else
    mkdir -p $BUILD_WORKING_DIR
fi



OPENCV_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-opencv_1.8.0
MLIR_SDK_ROOT=$CVI_TDL_ROOT/sophon_sdk/libsophon-0.4.10
MIDDLEWARE_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-ffmpeg_1.8.0
ISP_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-soc-libisp_1.0.0

if [[ "$1" == "Release" ]]; then
    BUILD_TYPE=Release
else
    BUILD_TYPE=SDKRelease
fi

BUILD_TYPE=Debug

FTP_SERVER_IP=${FTP_SERVER_IP:-10.80.0.5}

CONFIG_DUAL_OS="${CONFIG_DUAL_OS:-OFF}"
if [[ "$CONFIG_DUAL_OS" == "y" ]]; then
    CONFIG_DUAL_OS="ON"
fi

CLEAN_BUILD="${CLEAN_BUILD:-false}"
if [ "$CLEAN_BUILD" = "true" ]; then
    echo "Clean build requested. Cleaning up build directory..."
    rm -rf $BUILD_WORKING_DIR
    mkdir -p $BUILD_WORKING_DIR
else
    if [ ! -d "${BUILD_WORKING_DIR}" ]; then
        echo "Build directory doesn't exist. Creating..."
        mkdir -p $BUILD_WORKING_DIR
    else
        echo "Using existing build directory for incremental build..."
    fi
fi
pushd $BUILD_WORKING_DIR

# Check cmake version
CMAKE_VERSION="$(cmake --version | grep 'cmake version' | sed 's/cmake version //g')"
CMAKE_REQUIRED_VERSION="3.18.4"
echo "Checking cmake..."
if [ "$(printf '%s\n' "$CMAKE_REQUIRED_VERSION" "$CMAKE_VERSION" | sort -V | head -n1)" = "$CMAKE_REQUIRED_VERSION" ]; then
    echo "Current cmake version is ${CMAKE_VERSION}, satisfy the required version ${CMAKE_REQUIRED_VERSION}"
    CMAKE_BIN=$(which cmake)
else
    echo "Cmake minimum required version is ${CMAKE_REQUIRED_VERSION}, trying to download from ftp."
    if [ ! -f cmake-3.18.4-Linux-x86_64.tar.gz ]; then
        wget ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/third_party/cmake/cmake-3.18.4-Linux-x86_64.tar.gz
    fi
    tar zxf cmake-3.18.4-Linux-x86_64.tar.gz
    CMAKE_BIN=$PWD/cmake-3.18.4-Linux-x86_64/bin/cmake
fi

if [[ "$SDK_VER" == "uclibc" ]]; then
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain-uclibc-linux.cmake
    SYSTEM_PROCESSOR=ARM
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm/usr
elif [[ "$SDK_VER" == "32bit" ]]; then
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain-gnueabihf-linux.cmake
    SYSTEM_PROCESSOR=ARM
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm/usr
elif [[ "$SDK_VER" == "64bit" ]]; then
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain-aarch64-linux.cmake
    if [[ "$CROSS_COMPILE" == "aarch64-linux-" ]]; then
        TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain930-aarch64-linux.cmake
    fi
    SYSTEM_PROCESSOR=ARM64
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm64/usr
elif [[ "$SDK_VER" == "glibc_riscv64" ]]; then
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain-riscv64-linux.cmake
    SYSTEM_PROCESSOR=RISCV
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/riscv/usr/
elif [[ "$SDK_VER" == "musl_riscv64" ]]; then
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/riscv/usr/
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain-riscv64-musl.cmake
    SYSTEM_PROCESSOR=RISCV
else
    echo "Wrong SDK_VER=$SDK_VER"
    exit 1
fi




$CMAKE_BIN -G Ninja $CVI_TDL_ROOT \
        -DCVI_PLATFORM=$CHIP_ARCH \
        -DCVI_SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DENABLE_CVI_TDL_CV_UTILS=ON \
        -DMLIR_SDK_ROOT=$MLIR_SDK_ROOT \
        -DMIDDLEWARE_SDK_ROOT=$MIDDLEWARE_ROOT_DIR \
        -DISP_ROOT_DIR=$ISP_ROOT_DIR \
        -DOPENCV_ROOT_DIR=$OPENCV_ROOT_DIR \
        -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
        -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
        -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
        -DKERNEL_ROOT=$KERNEL_ROOT \
        -DUSE_TPU_IVE=$USE_TPU_IVE \
        -DMW_VER=$MW_VER \
        -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR \
        -DCONFIG_DUAL_OS=$CONFIG_DUAL_OS \
        -DFTP_SERVER_IP=$FTP_SERVER_IP

ninja -j8 || exit 1
ninja install || exit 1
popd
exit 0

