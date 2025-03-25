#!/bin/bash

# print usage
print_usage() {
    echo "Usage: ${BASH_SOURCE[0]} [options]"
    echo "Options:"
    echo "  CV181X         Build soph-pi 181X"
    echo "  CV186X         Build 186X"
    echo "  BM1688         Build BM1688 edge"
    echo "  BM1684X        Build BM1684X edge"
    echo "  CMODEL_CVITEK   Build linux x86_64"
    echo "  sample         Build samples only"
    echo "  all            Build both modules and sample"
    echo "  clean          Clean build"
}

# Check parameter
if [ "$#" -gt 2 ]; then
    echo "Error: Too many arguments"
    print_usage
    exit 1
fi

if [ -z "$TOP_DIR" ]; then
    TOP_DIR=$(cd $(dirname $0);cd ..; pwd)
fi

if [ -f "${TOP_DIR}/tdl_sdk/scripts/credential.sh" ]; then
  source "$TOP_DIR/tdl_sdk/scripts/credential.sh"
fi

# get tdl_sdk root dir
CVI_TDL_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Handle platform-specific build commands
if [[ "$1" == "CV181X" ]]; then
    echo "Building for CV181X platform..."
    export CHIP_ARCH=CV181X
    
    # Execute CV181X specific commands
    cd ..
    source build/envsetup_soc.sh
    defconfig sg2002_wevb_riscv64_sd
    export TPU_REL=1
    clean_all
    build_all
    cd tdl_sdk
    exit 0

elif [[ "$1" == "CV186X" ]]; then
    echo "Building for CV186X platform..."
    export CHIP_ARCH=CV186X
    
    # Execute CV186X specific commands
    cd ..
    source build/envsetup_soc.sh
    defconfig device_wevb_emmc
    export TPU_REL=1
    clean_device_all
    build_device_all
    cd tdl_sdk
    exit 0

elif [[ "$1" == "BM1688" ]]; then
    echo "Building for BM1688 platform..."
    export CHIP_ARCH=BM1688
    # Continue with the regular build process below
elif [[ "$1" == "BM1684X" ]]; then
    echo "Building for BM1684X platform..."
    export CHIP_ARCH=BM1684X
    # Continue with the regular build process below
elif [[ "$1" == "CMODEL_CVITEK" ]]; then

    if [ -e "dependency/CMODEL_CVITEK" ]; then
        echo "Building for CMODEL_CVITEK platform..."
        export CHIP_ARCH=CMODEL_CVITEK
    else
        echo "CMODEL_CVITEK not found"
        exit 1
    fi

elif [[ "$1" == "clean" ]]; then
    echo "Using ./${BASH_SOURCE[0]} clean"
    echo "Cleaning build..."
    
    # Set CHIP_ARCH if it's not already set
    if [ -z "$CHIP_ARCH" ]; then
        echo "CHIP_ARCH not set, cleaning all architectures"
        # Clean all possible build directories
        for arch in CV181X CV186X BM1688; do
            BUILD_WORKING_DIR="${CVI_TDL_ROOT}"/build_${arch}
            TDL_SDK_INSTALL_PATH="${CVI_TDL_ROOT}"/install/"${arch}"
            
            if [ -d "${BUILD_WORKING_DIR}" ]; then
                echo "Cleanup tmp folder for ${arch}."
                rm -rf ${BUILD_WORKING_DIR}
            fi
            if [ -d "${TDL_SDK_INSTALL_PATH}" ]; then
                echo "Cleanup install folder for ${arch}."
                rm -rf ${TDL_SDK_INSTALL_PATH}
            fi
        done
    else
        BUILD_WORKING_DIR="${CVI_TDL_ROOT}"/build_${CHIP_ARCH}
        TDL_SDK_INSTALL_PATH="${CVI_TDL_ROOT}"/install/"${CHIP_ARCH}"
        
        if [ -d "${BUILD_WORKING_DIR}" ]; then
            echo "Cleanup tmp folder for ${CHIP_ARCH}."
            rm -rf ${BUILD_WORKING_DIR}
        fi
        if [ -d "${TDL_SDK_INSTALL_PATH}" ]; then
            echo "Cleanup install folder for ${CHIP_ARCH}."
            rm -rf ${TDL_SDK_INSTALL_PATH}
        fi
    fi
    exit 0

elif [[ "$1" == "sample" ]]; then
    echo "Using ./${BASH_SOURCE[0]} sample"
    echo "Compiling sample..."
    BUILD_OPTION=sample
    
    # Check if CHIP_ARCH is set
    if [ -z "$CHIP_ARCH" ]; then
        echo "Error: CHIP_ARCH not set. Please specify a platform (cv181x, cv186x, bm168x) first."
        print_usage
        exit 1
    fi
    
elif [[ "$1" == "all" ]]; then
    echo "Using ./${BASH_SOURCE[0]} all"
    echo "Compiling modules and sample..."

    cd ..
    source build/envsetup_soc.sh
    oldconfig
    cd tdl_sdk

    BUILD_OPTION=all
    
    # Check if CHIP_ARCH is set
    if [ -z "$CHIP_ARCH" ]; then
        echo "Error: CHIP_ARCH not set. Please specify a platform (cv181x, cv186x, bm168x) first."
        print_usage
        exit 1
    fi
    
elif [ "$#" -eq 0 ]; then
    echo "Using ./${BASH_SOURCE[0]}"
    echo "Compiling modules..."
    
    # Check if CHIP_ARCH is set
    if [ -z "$CHIP_ARCH" ]; then
        echo "Error: CHIP_ARCH not set. Please specify a platform (cv181x, cv186x, bm168x) first."
        print_usage
        exit 1
    fi
    
else
    echo "Error: Invalid option"
    print_usage
    exit 1
fi

# set build working dir
BUILD_WORKING_DIR="${CVI_TDL_ROOT}"/build/${CHIP_ARCH}
mkdir -p ${BUILD_WORKING_DIR}
BUILD_DOWNLOAD_DIR="${BUILD_WORKING_DIR}"/_deps

# set install path
TDL_SDK_INSTALL_PATH="${CVI_TDL_ROOT}"/install/"${CHIP_ARCH}"

# Set build option and type
BUILD_TYPE=Debug

# check system type
CONFIG_DUAL_OS=OFF
if [ -n "${ALIOS_PATH}" ]; then
    CONFIG_DUAL_OS=ON
fi

if [[ "$CHIP_ARCH" == "BM1688" ]]; then
    CROSS_COMPILE_PATH=$CVI_TDL_ROOT/../host-tools/gcc/gcc-buildroot-9.3.0-aarch64-linux-gnu/
    CROSS_COMPILE=aarch64-linux-
    CV_UTILS=OFF
    OPENCV_ROOT_DIR=$CVI_TDL_ROOT/dependency/BM1688/sophon-opencv
    TPU_SDK_INSTALL_PATH=$CVI_TDL_ROOT/dependency/BM1688/libsophon
    MPI_PATH=$CVI_TDL_ROOT/dependency/BM1688/sophon-ffmpeg
    ISP_ROOT_DIR=$CVI_TDL_ROOT/dependency/BM1688/sophon-soc-libisp
    
elif [[ "$CHIP_ARCH" == "BM1684X" ]]; then
    CROSS_COMPILE_PATH=$CVI_TDL_ROOT/../host-tools/gcc/gcc-buildroot-9.3.0-aarch64-linux-gnu/
    CROSS_COMPILE=aarch64-linux-
    CV_UTILS=OFF
    OPENCV_ROOT_DIR=$CVI_TDL_ROOT/dependency/BM1684X/sophon-opencv
    TPU_SDK_INSTALL_PATH=$CVI_TDL_ROOT/dependency/BM1684X/libsophon
    MPI_PATH=$CVI_TDL_ROOT/dependency/BM1684X/sophon-ffmpeg

elif [[ "$CHIP_ARCH" == "CMODEL_CVITEK" ]]; then
    CROSS_COMPILE_PATH=/usr/
    CROSS_COMPILE=""
    CV_UTILS=OFF
    OPENCV_ROOT_DIR=$CVI_TDL_ROOT/dependency/CMODEL_CVITEK/opencv
    TPU_SDK_INSTALL_PATH=$CVI_TDL_ROOT/dependency/CMODEL_CVITEK

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
elif [[ "${CHIP_ARCH}" == "BM1684X" ]]; then
    USE_TPU_IVE=OFF
elif [[ "${CHIP_ARCH}" == "CMODEL_CVITEK" ]]; then
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
