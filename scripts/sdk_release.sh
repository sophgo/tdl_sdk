#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVI_TDL_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$CVI_TDL_ROOT/tmp
BUILD_WORKING_DIR=$TMP_WORKING_DIR/build_sdk
BUILD_DOWNLOAD_DIR=$TMP_WORKING_DIR/_deps

if [[ "$1" == "Asan" ]]; then
    BUILD_TYPE=Asan
else
    BUILD_TYPE=SDKRelease
fi

if [ "${FTP_SERVER_IP}" = "" ]; then
    FTP_SERVER_IP=10.80.0.5/sw_rls
fi

if [[ "$CONFIG_DUAL_OS" == "y" ]]; then
    CONFIG_DUAL_OS="ON"
else
    CONFIG_DUAL_OS="OFF"
fi

REPO_USER=""
CURRENT_USER="$(git config user.name)"
if [[ "${CURRENT_USER}" != "sw_jenkins" ]]; then
REPO_USER="$(git config user.name)@"
fi
echo "repo user : $REPO_USER"

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
    if [ ! -f cmake-3.18.4-Linux-x86_64.tar.gz ]; then
        wget ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/cmake/cmake-3.18.4-Linux-x86_64.tar.gz
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

if [[ "$CHIP_ARCH" == "CV183X" ]]; then
    # 3X use tpu opencv
    SHRINK_OPENCV_SIZE=OFF
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "CV182X" ]]; then
    SHRINK_OPENCV_SIZE=ON
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "CV181X" ]]; then
    USE_TPU_IVE=OFF
    SHRINK_OPENCV_SIZE=ON
elif [[ "$CHIP_ARCH" == "CV180X" ]]; then
    SHRINK_OPENCV_SIZE=ON
    USE_TPU_IVE=ON
elif [[ "$CHIP_ARCH" == "ATHENA2" ]]; then
    CHIP_ARCH=CV186X
    SHRINK_OPENCV_SIZE=OFF
    USE_TPU_IVE=OFF
else
    echo "Unsupported chip architecture: ${CHIP_ARCH}"
    exit 1
fi

$CMAKE_BIN -G Ninja $CVI_TDL_ROOT -DCVI_PLATFORM=$CHIP_ARCH \
                                        -DCVI_SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR \
                                        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                                        -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                                        -DENABLE_CVI_TDL_CV_UTILS=ON \
                                        -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                                        -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                                        -DSYSTEM_OUT_DIR=$SYSTEM_OUT_DIR \
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
                                        -DBUILD_DOWNLOAD_DIR=$BUILD_DOWNLOAD_DIR \
                                        -DREPO_USER=$REPO_USER \
                                        -DCONFIG_DUAL_OS=$CONFIG_DUAL_OS

ninja -j8 || exit 1
ninja install || exit 1
popd

echo "trying to build sample in released folder."
if [[ "$CHIP_ARCH" != "CV180X" ]]; then
  pushd ${AI_SDK_INSTALL_PATH}/sample/cvi_tdl
  make KERNEL_ROOT="$KERNEL_ROOT" MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE="$USE_TPU_IVE" SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR CHIP=$CHIP_ARCH -j10 || exit 1
  make clean || exit 1
  echo "done"
  popd

  pushd ${AI_SDK_INSTALL_PATH}/sample/cvi_tdl_app
  make KERNEL_ROOT="$KERNEL_ROOT" MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE=$USE_TPU_IVE SYSTEM_OUT_DIR=$SYSTEM_OUT_DIR CHIP=$CHIP_ARCH SDK_VER=$SDK_VER SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR -j10 || exit 1
  make clean || exit 1
  echo "done"
  popd

  pushd ${AI_SDK_INSTALL_PATH}/sample/cvi_md
  make KERNEL_ROOT="$KERNEL_ROOT" MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE=$USE_TPU_IVE SYSTEM_OUT_DIR=$SYSTEM_OUT_DIR CHIP=$CHIP_ARCH SDK_VER=$SDK_VER SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR -j10 || exit 1
  make clean || exit 1
  echo "done"
  popd

  pushd ${AI_SDK_INSTALL_PATH}/sample/cvi_preprocess
  make KERNEL_ROOT="$KERNEL_ROOT" MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE=$USE_TPU_IVE SYSTEM_OUT_DIR=$SYSTEM_OUT_DIR CHIP=$CHIP_ARCH SDK_VER=$SDK_VER SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR -j10 || exit 1
  make clean || exit 1
  echo "done"
  popd

  pushd ${AI_SDK_INSTALL_PATH}/sample/cvi_draw_rect
  make KERNEL_ROOT="$KERNEL_ROOT" W_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" USE_TPU_IVE=$USE_TPU_IVE SYSTEM_OUT_DIR=$SYSTEM_OUT_DIR CHIP=$CHIP_ARCH SDK_VER=$SDK_VER SYSTEM_PROCESSOR=$SYSTEM_PROCESSOR -j10 || exit 1
  make clean || exit 1
  echo "done"
  popd

fi

rm -rf ${AI_SDK_INSTALL_PATH}/sample/tmp_install

#wget -nc -P ${AI_SDK_INSTALL_PATH}/doc https://doc.sophgo.com/cvitek-develop-docs/master/docs_latest_release/CV180x_CV181x/zh/01.software/TPU/AI_SDK_Software_Development_Guide/build/AISDKSoftwareDevelopmentGuide_zh.pdf

#wget -nc -P ${AI_SDK_INSTALL_PATH}/doc http://doc.sophgo.com/cvitek-develop-docs/master/docs_latest_release/CV180x_CV181x/en/01.software/TPU/AI_SDK_Software_Development_Guide/build/AISDKSoftwareDevelopmentGuide_en.pdf

#wget -nc -P ${AI_SDK_INSTALL_PATH}/doc http://doc.sophgo.com/cvitek-develop-docs/master/docs_latest_release/CV180x_CV181x/en/01.software/TPU/YOLO_Development_Guide/build/YOLODevelopmentGuide_en.pdf

#wget -nc -P ${AI_SDK_INSTALL_PATH}/doc http://doc.sophgo.com/cvitek-develop-docs/master/docs_latest_release/CV180x_CV181x/zh/01.software/TPU/YOLO_Development_Guide/build/YOLODevelopmentGuide_zh.pdf
