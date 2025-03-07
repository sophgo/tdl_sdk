#!/bin/bash

# print usage
print_usage() {
    echo "Usage: source ${BASH_SOURCE[0]} [options]"
    echo "Options:"
    echo "  (no option) Build modules only"
    echo "  sample      Build samples only"
    echo "  all         Build both modules and sample"
    echo "  bm168x      Build BM168X edge"
    echo "  clean       Clean build"
}

# Check parameter
if [ "$#" -gt 2 ]; then
    echo "Error: Too many arguments"
    print_usage
    exit 1
fi

if [[ "$1" == "bm168x" ]]; then
    CHIP_ARCH=BM1688
fi

if [ -f "${TOP_DIR}/tdl_sdk/scripts/credential.sh" ]; then
  source "$TOP_DIR/tdl_sdk/scripts/credential.sh"
fi

# get tdl_sdk root dir
CVI_TDL_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# set build working dir
BUILD_WORKING_DIR="${CVI_TDL_ROOT}"/build_${CHIP_ARCH}
BUILD_DOWNLOAD_DIR="${BUILD_WORKING_DIR}"/_deps

# set install path
TDL_SDK_INSTALL_PATH="${CVI_TDL_ROOT}"/install/"${CHIP_ARCH}"

# Set build option and type
BUILD_TYPE=Debug

# print building message
if [ "$#" -eq 0 ]; then
    echo "Using ./${BASH_SOURCE[0]}"
    echo "Compiling modules..."
elif [ "$1" = "sample" ]; then
    echo "Using ./${BASH_SOURCE[0]} sample"
    echo "Compiling sample..."
    BUILD_OPTION=sample
elif [ "$1" = "all" ]; then
    echo "Using ./${BASH_SOURCE[0]} all"
    echo "Compiling modules and sample..."
    BUILD_OPTION=all
elif [ "$1" = "bm168x" ]; then
    echo "Using ./${BASH_SOURCE[0]} bm168x"
    echo "Compiling BM168X edge..."
    BUILD_OPTION=bm168x
elif [ "$1" = "clean" ]; then
    echo "Using ./${BASH_SOURCE[0]} clean"
    echo "Cleaning build..."
    if [ -d "${BUILD_WORKING_DIR}" ]; then
        echo "Cleanup tmp folder."
        rm -rf ${BUILD_WORKING_DIR}
    fi
    if [ -d "${TDL_SDK_INSTALL_PATH}" ]; then
        echo "Cleanup install folder."
        rm -rf ${TDL_SDK_INSTALL_PATH}
    fi
    exit 0
else
    echo "Error: Invalid option"
    print_usage
    exit 1
fi

# check system type
CONFIG_DUAL_OS=OFF
if [ -n "${ALIOS_PATH}" ]; then
    CONFIG_DUAL_OS=ON
fi


if [[ "$CHIP_ARCH" == "BM1688" ]]; then
    CROSS_COMPILE_PATH=$CVI_TDL_ROOT/../host-tools/gcc/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/
    CROSS_COMPILE=aarch64-none-linux-gnu-
    CV_UTILS=OFF
    OPENCV_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-opencv_1.8.0
    TPU_SDK_INSTALL_PATH=$CVI_TDL_ROOT/sophon_sdk/libsophon-0.4.10
    MPI_PATH=$CVI_TDL_ROOT/sophon_sdk/sophon-ffmpeg_1.8.0
    ISP_ROOT_DIR=$CVI_TDL_ROOT/sophon_sdk/sophon-soc-libisp_1.0.0

else
    CV_UTILS=ON
    TPU_SDK_INSTALL_PATH="$OUTPUT_DIR"/tpu_"$SDK_VER"/cvitek_tpu_sdk
    IVE_SDK_INSTALL_PATH="$OUTPUT_DIR"/tpu_"$SDK_VER"/cvitek_ive_sdk
    MW_VER=v2
    MPI_PATH="${TOP_DIR}"/cvi_mpi
fi

# set host-tool
HOST_TOOL_PATH="${CROSS_COMPILE_PATH}"
TARGET_MACHINE="$(${CROSS_COMPILE_PATH}/bin/${CROSS_COMPILE}gcc -dumpmachine)"
TOOLCHAIN_FILE="${CVI_TDL_ROOT}"/toolchain/"${TARGET_MACHINE}".cmake

if [ -d "${BUILD_WORKING_DIR}" ]; then
    echo "BUILD_WORKING_DIR=${BUILD_WORKING_DIR}"
else
    mkdir -p ${BUILD_WORKING_DIR}
fi

# into tmp/build_sdk
pushd ${BUILD_WORKING_DIR}

CMAKE_BIN=$(command -v cmake)

# check if use TPU_IVE
if [[ "${CHIP_ARCH}" == "CV183X" ]]; then
    USE_TPU_IVE=ON
elif [[ "${CHIP_ARCH}" == "CV182X" ]]; then
    USE_TPU_IVE=ON
elif [[ "${CHIP_ARCH}" == "CV181X" ]]; then
    USE_TPU_IVE=OFF
elif [[ "${CHIP_ARCH}" == "CV180X" ]]; then
    USE_TPU_IVE=ON
elif [[ "${CHIP_ARCH}" == "SOPHON" ]]; then
    MPI_PATH="${TOP_DIR}"/middleware/"${MW_VER}"
    USE_TPU_IVE=OFF
elif [[ "${CHIP_ARCH}" == "BM1688" ]]; then
    USE_TPU_IVE=OFF
else
    echo "Unsupported chip architecture: ${CHIP_ARCH}"
    exit 1
fi

# build start
$CMAKE_BIN -G Ninja ${CVI_TDL_ROOT} -DCVI_PLATFORM=${CHIP_ARCH} \
                                    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
                                    -DENABLE_CVI_TDL_CV_UTILS=${CV_UTILS} \
                                    -DMLIR_SDK_ROOT=${TPU_SDK_INSTALL_PATH} \
                                    -DISP_ROOT_DIR=${ISP_ROOT_DIR} \
                                    -DOPENCV_ROOT_DIR=${OPENCV_ROOT_DIR} \
                                    -DMIDDLEWARE_SDK_ROOT=${MPI_PATH} \
                                    -DTPU_IVE_SDK_ROOT=${IVE_SDK_INSTALL_PATH} \
                                    -DCMAKE_INSTALL_PREFIX=${TDL_SDK_INSTALL_PATH} \
                                    -DTOOLCHAIN_ROOT_DIR=${HOST_TOOL_PATH} \
                                    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
                                    -DUSE_TPU_IVE=${USE_TPU_IVE} \
                                    -DBUILD_DOWNLOAD_DIR=${BUILD_DOWNLOAD_DIR} \
                                    -DCONFIG_DUAL_OS=${CONFIG_DUAL_OS} \
                                    -DBUILD_OPTION=${BUILD_OPTION} \
                                    -DTARGET_MACHINE=${TARGET_MACHINE} \
                                    -DMW_VER=${MW_VER} \
                                    -DFTP_SERVER_IP=${FTP_SERVER_IP} \
                                    -DFTP_SERVER_NAME=${FTP_SERVER_NAME} \
                                    -DFTP_SERVER_PWD=${FTP_SERVER_PWD}

test $? -ne 0 && echo "cmake tdl_sdk failed !!" && popd && exit 1

ninja -j8 || exit 1
ninja install || exit 1
popd
# build end
