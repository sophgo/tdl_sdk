#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVI_TDL_ROOT=$(readlink -f $SCRIPT_DIR/../)
SDK_VER=64bit

BUILD_WORKING_DIR=$CVI_TDL_ROOT/build1688
BUILD_DOWNLOAD_DIR=$BUILD_WORKING_DIR/_deps
AI_SDK_INSTALL_PATH=$BUILD_WORKING_DIR/install
CHIP_ARCH=BM1688
USE_TPU_IVE=OFF


HOST_TOOL_PATH=/data/fuquan.ke/nfsuser/cv_bm_dev/sdk_package/host-tools/gcc/gcc-buildroot-9.3.0-aarch64-linux-gnu


if [ ! -d "$BUILD_WORKING_DIR" ]; then
    echo "BUILD_WORKING_DIR=$BUILD_WORKING_DIR"
    mkdir -p $BUILD_WORKING_DIR
fi

if [ -d "$AI_SDK_INSTALL_PATH" ]; then
    echo "AI_SDK_INSTALL_PATH=$AI_SDK_INSTALL_PATH"
else
    mkdir -p $AI_SDK_INSTALL_PATH
fi

if [ -d "$BUILD_DOWNLOAD_DIR" ]; then
    echo "BUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR"
else
    mkdir -p $BUILD_DOWNLOAD_DIR
fi



OPENCV_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-opencv_1.8.0
MLIR_SDK_ROOT=$CVI_TDL_ROOT/sophon_sdk/libsophon-0.4.10
MIDDLEWARE_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-ffmpeg_1.8.0
ISP_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-soc-libisp_1.0.0
TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain930-aarch64-linux.cmake

if [[ "$1" == "Release" ]]; then
    BUILD_TYPE=Release
else
    BUILD_TYPE=SDKRelease
fi

BUILD_TYPE=Debug

FTP_SERVER_IP=${FTP_SERVER_IP:-10.80.0.5}
echo "FTP_SERVER_IP=$FTP_SERVER_IP"
CONFIG_DUAL_OS="${CONFIG_DUAL_OS:-OFF}"
if [[ "$CONFIG_DUAL_OS" == "y" ]]; then
    CONFIG_DUAL_OS="ON"
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


echo "CMAKE_BIN=$CMAKE_BIN "
echo "TOOLCHAIN_FILE=$TOOLCHAIN_FILE"

echo "当前工作目录：$(pwd)"
# $CMAKE_BIN .. \
#         -DCVI_PLATFORM=$CHIP_ARCH \
#         -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
#         -DENABLE_CVI_TDL_CV_UTILS=ON \
#         -DMLIR_SDK_ROOT=$MLIR_SDK_ROOT \
#         -DMIDDLEWARE_SDK_ROOT=$MIDDLEWARE_ROOT_DIR \
#         -DISP_ROOT_DIR=$ISP_ROOT_DIR \
#         -DOPENCV_ROOT_DIR=$OPENCV_ROOT_DIR \
#         -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
#         -DUSE_TPU_IVE=$USE_TPU_IVE \
#         -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR \
#         -DCONFIG_DUAL_OS=$CONFIG_DUAL_OS \
#         -DFTP_SERVER_IP=$FTP_SERVER_IP
#         # -DCMAKE_C_COMPILER=$C_COMPILER \
#         # -DCMAKE_CXX_COMPILER=$CXX_COMPILER

# make -j8

# make install


$CMAKE_BIN -G Ninja $CVI_TDL_ROOT \
        -DCVI_PLATFORM=$CHIP_ARCH \
        -DCVI_SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
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



