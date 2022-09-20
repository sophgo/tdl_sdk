#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVIAI_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$CVIAI_ROOT/tmp
BUILD_WORKING_DIR=$TMP_WORKING_DIR/build_sdk
BUILD_DOWNLOAD_DIR=$TMP_WORKING_DIR/_deps

if [[ "$1" == "Asan" ]]; then
    BUILD_TYPE=Asan
else
    BUILD_TYPE=SDKRelease
fi

if [ -d "${BUILD_WORKING_DIR}" ]; then
    echo "Cleanup tmp folder."
    rm -rf $BUILD_WORKING_DIR
fi

echo "Creating tmp working directory."
mkdir -p $BUILD_WORKING_DIR
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
    wget -c ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/cmake/cmake-3.18.4-Linux-x86_64.tar.gz
    tar zxf cmake-3.18.4-Linux-x86_64.tar.gz
    CMAKE_BIN=$PWD/cmake-3.18.4-Linux-x86_64/bin/cmake
fi

if [[ "$SDK_VER" == "uclibc" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-uclibc-linux.cmake
    SYSTEM_PROCESSOR=ARM
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm/usr
elif [[ "$SDK_VER" == "32bit" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-gnueabihf-linux.cmake
    SYSTEM_PROCESSOR=ARM
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm/usr
elif [[ "$SDK_VER" == "64bit" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-aarch64-linux.cmake
    SYSTEM_PROCESSOR=ARM64
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/arm64/usr
elif [[ "$SDK_VER" == "glibc_riscv64" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-riscv64-linux.cmake
    SYSTEM_PROCESSOR=RISCV
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/riscv/usr/
elif [[ "$SDK_VER" == "musl_riscv64" ]]; then
    KERNEL_ROOT="${KERNEL_PATH}"/build/"${PROJECT_FULLNAME}"/riscv/usr/
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-riscv64-musl.cmake
    SYSTEM_PROCESSOR=RISCV
else
    echo "Wrong SDK_VER=$SDK_VER"
    exit 1
fi

if [ "${PHOBOS_SIMULATE}" != "" ]; then
    CHIP_ARCH=MARS_PHOBOS
fi

if [[ "$CHIP_ARCH" == "CV183X" ]]; then
    SHRINK_OPENCV_SIZE=OFF
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "CV182X" ]]; then
    SHRINK_OPENCV_SIZE=ON
    OPENCV_INSTALL_PATH=""
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "MARS" ]]; then
    USE_TPU_IVE=OFF
    IVE_SDK_INSTALL_PATH=""
    SHRINK_OPENCV_SIZE=ON
elif [[ "$CHIP_ARCH" == "PHOBOS" ]]; then
    SHRINK_OPENCV_SIZE=ON
    OPENCV_INSTALL_PATH=""
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "MARS_PHOBOS" ]]; then
    USE_TPU_IVE=ON
    SHRINK_OPENCV_SIZE=ON
else
    echo "Unsupported chip architecture: ${CHIP_ARCH}"
    exit 1
fi

$CMAKE_BIN -G Ninja $CVIAI_ROOT -DCVI_PLATFORM=$CHIP_ARCH \
                                        -DCVI_SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR \
                                        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                                        -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                                        -DENABLE_CVIAI_CV_UTILS=ON \
                                        -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                                        -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                                        -DTPU_IVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
                                        -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
                                        -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
                                        -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
                                        -DSHRINK_OPENCV_SIZE=$SHRINK_OPENCV_SIZE \
                                        -DKERNEL_ROOT=$KERNEL_ROOT \
                                        -DUSE_TPU_IVE=$USE_TPU_IVE \
                                        -DMW_VER=$MW_VER \
                                        -DCVI_MIDDLEWARE_3RD_LDFLAGS="$CVI_TARGET_PACKAGES_LIBDIR" \
                                        -DCVI_MIDDLEWARE_3RD_INCCLAGS="$CVI_TARGET_PACKAGES_INCLUDE" \
                                        -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR


ninja -j8 || exit 1
ninja install || exit 1
popd

echo "trying to build sample in released folder."
for v in $CVI_TARGET_PACKAGES_LIBDIR; do
    if [[ "$v" == *"cvitracer"* ]]; then
        CVI_TRACER_LIB_PATH=${v:2:$((${#v}-2))}
        searchstring="cvitracer"
        rest=${CVI_TRACER_LIB_PATH#*$searchstring}
        index=$((${#CVI_TRACER_LIB_PATH} - ${#rest} - ${#searchstring}))
        CVI_TRACER_ROOT_PATH=${CVI_TRACER_LIB_PATH:0:$index}cvitracer
        break
    fi
done

if [ "$CHIP_ARCH" != "PHOBOS" ] && [ "$CHIP_ARCH" != "MARS_PHOBOS" ]; then

  pushd ${AI_SDK_INSTALL_PATH}/module/app
  make MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE="$USE_TPU_IVE" SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR CHIP=$CHIP_ARCH -j10 || exit 1
  make install || exit 1
  make clean || exit 1
  echo "done"
  popd

  pushd ${AI_SDK_INSTALL_PATH}/sample
  make MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" CVI_TRACER_PATH="$CVI_TRACER_ROOT_PATH" USE_TPU_IVE=$USE_TPU_IVE CHIP=$CHIP_ARCH SDK_VER=$SDK_VER SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR -j10 || exit 1
  make install || exit 1
  make clean || exit 1
  echo "done"
  popd
fi

rm -rf ${AI_SDK_INSTALL_PATH}/tmp_install
