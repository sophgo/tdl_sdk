#!/bin/bash

# print usage
print_usage() {
    echo "Usage: source ${BASH_SOURCE[0]} build_type"
    echo "build_type:"
    echo "  all: Build 181x and a2"
    echo "  BM1688: Build BM1688 "
}

# Check parameter
if [ "$1" != "all" -a "$1" != "BM1688" ]; then
    echo "Error  arguments"
    print_usage
    exit 1
fi

if [[ "$1" == "BM1688" ]]; then
    CHIP_ARCH=$1
fi


# get tdl_sdk root dir
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVI_TDL_ROOT=$(readlink -f $SCRIPT_DIR)

BUILD_WORKING_DIR=$CVI_TDL_ROOT/build$CHIP_ARCH
BUILD_DOWNLOAD_DIR="$BUILD_WORKING_DIR"/_deps
TDL_SDK_INSTALL_PATH="$CVI_TDL_ROOT"/install/"$CHIP_ARCH"


# Set build opetion and type
BUILD_TYPE=Debug


if [ -d "$BUILD_DOWNLOAD_DIR" ]; then
    echo "BUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR"
else
    mkdir -p $BUILD_DOWNLOAD_DIR
fi

if [ -d "$TDL_SDK_INSTALL_PATH" ]; then
    echo "TDL_SDK_INSTALL_PATH=$TDL_SDK_INSTALL_PATH"
else
    mkdir -p $TDL_SDK_INSTALL_PATH
fi

# check system type
CONFIG_DUAL_OS=OFF
if [ -n "$ALIOS_PATH" ]; then
    CONFIG_DUAL_OS=ON
fi

# set ftp server
FTP_SERVER_IP=${FTP_SERVER_IP:-10.80.0.5}

echo $CHIP_ARCH
if [[ "$CHIP_ARCH" == "BM1688" ]]; then

    HOST_TOOL_PATH=$CVI_TDL_ROOT/../host-tools/gcc/gcc-buildroot-9.3.0-aarch64-linux-gnu

    OPENCV_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-opencv_1.8.0
    MLIR_SDK_ROOT=$CVI_TDL_ROOT/sophon_sdk/libsophon-0.4.10
    MIDDLEWARE_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-ffmpeg_1.8.0
    ISP_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-soc-libisp_1.0.0
    TOOLCHAIN_FILE=$CVI_TDL_ROOT/toolchain/toolchain930-aarch64-linux.cmake

else
    HOST_TOOL_PATH="$CROSS_COMPILE_PATH"
    TPU_SDK_INSTALL_PATH="$OUTPUT_DIR"/tpu_"$SDK_VER"/cvitek_tpu_sdk
    IVE_SDK_INSTALL_PATH="$OUTPUT_DIR"/tpu_"$SDK_VER"/cvitek_ive_sdk
    TARGET_MACHINE="$(${CROSS_COMPILE}gcc -dumpmachine)"
    TOOLCHAIN_FILE="$CVI_TDL_ROOT"/toolchain/"$TARGET_MACHINE".cmake

    MW_VER=v2
    MPI_PATH="$TOP_DIR"/cvi_mpi
    BUILD_OPTION=$1
fi


# into tmp/build_sdk
pushd $BUILD_WORKING_DIR

# Check cmake version
CMAKE_VERSION="$(cmake --version | grep 'cmake version' | sed 's/cmake version //g')"
CMAKE_REQUIRED_VERSION="3.18.4"
CMAKE_TAR="cmake-3.18.4-Linux-x86_64.tar.gz"
CMAKE_DOWNLOAD_URL="ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/third_party/cmake/${CMAKE_TAR}"
echo "Checking cmake..."
if [ "$(printf '%s\n' "$CMAKE_REQUIRED_VERSION" "$CMAKE_VERSION" | sort -V | head -n1)" = "$CMAKE_REQUIRED_VERSION" ]; then
    CMAKE_BIN=$(command -v cmake)
else
    echo "Cmake version need ${CMAKE_REQUIRED_VERSION}, trying to download from ftp."
    if [ ! -f "$CMAKE_TAR" ]; then
        wget "$CMAKE_DOWNLOAD_URL"
    fi
    tar -zxf $CMAKE_TAR
    CMAKE_BIN=$PWD/cmake-3.18.4-Linux-x86_64/bin/cmake
fi

# check if use TPU_IVE
if [[ "$CHIP_ARCH" == "CV183X" ]]; then
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "CV182X" ]]; then
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "CV181X" ]]; then
    USE_TPU_IVE=OFF
elif [[ "$CHIP_ARCH" == "CV180X" ]]; then
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "SOPHON" ]]; then
    MPI_PATH="$TOP_DIR"/middleware/"$MW_VER"
    USE_TPU_IVE=OFF
elif [[ "$CHIP_ARCH" == "BM1688" ]]; then
    USE_TPU_IVE=OFF
else
    echo "Unsupported chip architecture: ${CHIP_ARCH}"
    exit 1
fi


if [[ "$CHIP_ARCH" != "BM1688" ]]; then

    # build start
    $CMAKE_BIN -G Ninja $CVI_TDL_ROOT \
            -DCVI_PLATFORM=$CHIP_ARCH \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DENABLE_CVI_TDL_CV_UTILS=ON \
            -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
            -DMIDDLEWARE_SDK_ROOT=$MPI_PATH \
            -DTPU_IVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
            -DCMAKE_INSTALL_PREFIX=$TDL_SDK_INSTALL_PATH \
            -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
            -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
            -DKERNEL_ROOT=$KERNEL_ROOT \
            -DUSE_TPU_IVE=$USE_TPU_IVE \
            -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR \
            -DCONFIG_DUAL_OS=$CONFIG_DUAL_OS \
            -DBUILD_OPTION=$BUILD_OPTION \
            -DTARGET_MACHINE=$TARGET_MACHINE \
            -DMW_VER=$MW_VER \
            -DFTP_SERVER_IP=$FTP_SERVER_IP


else

    $CMAKE_BIN -G Ninja $CVI_TDL_ROOT \
            -DCVI_PLATFORM=$CHIP_ARCH \
            -DCVI_SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DMLIR_SDK_ROOT=$MLIR_SDK_ROOT \
            -DMIDDLEWARE_SDK_ROOT=$MIDDLEWARE_ROOT_DIR \
            -DISP_ROOT_DIR=$ISP_ROOT_DIR \
            -DOPENCV_ROOT_DIR=$OPENCV_ROOT_DIR \
            -DCMAKE_INSTALL_PREFIX=$TDL_SDK_INSTALL_PATH \
            -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
            -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
            -DKERNEL_ROOT=$KERNEL_ROOT \
            -DUSE_TPU_IVE=$USE_TPU_IVE \
            -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR \
            -DCONFIG_DUAL_OS=$CONFIG_DUAL_OS \
            -DFTP_SERVER_IP=$FTP_SERVER_IP

fi

test $? -ne 0 && echo "cmake tdl_sdk failed !!" && popd && exit 1

ninja -j8 || exit 1
ninja install || exit 1
popd
# build end

