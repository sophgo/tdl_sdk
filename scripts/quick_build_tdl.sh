#!/bin/bash

# print usage
print_usage() {
  echo "Usage: source ${BASH_SOURCE[0]} [option] [type]"
  echo "Options:"
  echo "  (no option) Build modules only"
  echo "  sample      Build samples only"
  echo "  all         Build both modules and sample"
}

# check parameter
if [ "$#" -gt 2 ]; then
  echo "Error: Too many arguments"
  print_usage
  return 1
fi

# set repo user
REPO_USER=""
CURRENT_USER="$(git config user.name)"
if [[ "${CURRENT_USER}" != "sw_jenkins" ]]; then
  REPO_USER="$(git config user.name)@"
fi
echo "repo user : ${REPO_USER}"

# get tdl_sdk root dir
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVI_TDL_ROOT=$(readlink -f ${SCRIPT_DIR}/../)
TOP_DIR=${TOP_DIR:-"$(readlink -f ${CVI_TDL_ROOT}/../)"}
export TOP_DIR=${TOP_DIR}

# set working dir
TMP_WORKING_DIR="${CVI_TDL_ROOT}"/tmp
BUILD_WORKING_DIR="${TMP_WORKING_DIR}"/build_sdk
BUILD_DOWNLOAD_DIR="${TMP_WORKING_DIR}"/_deps
TDL_SDK_INSTALL_PATH="${CVI_TDL_ROOT}"/install

# set dependency
MPI_PATH=${MPI_PATH:-"${TOP_DIR}/cvi_mpi"}
TPU_SDK_PATH=${TPU_SDK_INSTALL_PATH:-"${TOP_DIR}/cvitek_tpu_sdk"}
IVE_SDK_PATH=${IVE_SDK_INSTALL_PATH:-"${TOP_DIR}/cvitek_ive_sdk"}

# set build options
BUILD_TYPE=SDKRelease
BUILD_OPTION=

# set build opetion and type
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
elif [ "$1" = "clean" ]; then
  echo "Using ./${BASH_SOURCE[0]} clean"
  echo "Cleaning build..."
  if [ -d "${BUILD_DOWNLOAD_DIR}" ]; then
    echo "Cleanup tmp folder."
    rm -rf ${BUILD_DOWNLOAD_DIR}
  fi
  if [ -d "${TDL_SDK_INSTALL_PATH}" ]; then
    echo "Cleanup install folder."
    rm -rf ${TDL_SDK_INSTALL_PATH}
  fi
  if [ -d "${MPI_PATH}" ]; then
    echo "Cleanup cvi_mpi folder."
    rm -rf ${MPI_PATH}
  fi
  if [ -d "${TPU_SDK_PATH}" ]; then
    echo "Cleanup cvitek_tpu_sdk folder."
    rm -rf ${TPU_SDK_PATH}
  fi
  if [ -d "${IVE_SDK_PATH}" ]; then
    echo "Cleanup cvitek_ive_sdk folder."
    rm -rf ${IVE_SDK_PATH}
  fi
  if [ -d "${TOP_DIR}/cvi_rtsp" ]; then
    echo "Cleanup cvi_rtsp folder."
    rm -rf ${TOP_DIR}/cvi_rtsp
  fi
  exit 0
else
  echo "Error: Invalid option"
  print_usage
  exit 1
fi


# set host-tool
HOST_TOOL_PATH=${HOST_TOOL_PATH:-"${TOP_DIR}/host-tools"}
if [ -d "${HOST_TOOL_PATH}" ]; then
  echo "host-tools path : ${HOST_TOOL_PATH}"
else
  echo "No host-tools in ${TOP_DIR}, please check!"
  exit 1
fi

# export parameter
CHIP_ARCH=${CHIP_ARCH:-}
DUAL_OS=OFF
MW_VER=${MW_VER:-v2}
FTP_SERVER_IP=${FTP_SERVER_IP:-10.80.0.5}
SDK_VER=${SDK_VER:-}
ARM64_VER=${ARM64_VER:-}
if [ "${SDK_VER}" == "64bit" ]; then
  if [ "${TOOLCHAIN_GLIBC_ARM64_V930}" == "y" ]; then
    ARM64_VER=v930
    echo "aarch64 toolchain select gcc-buildroot-9.3.0-aarch64-linux-gnu !"
  elif [ "${TOOLCHAIN_GLIBC_ARM64_V1131}" == "y" ]; then
    ARM64_VER=v1131
    echo "aarch64 toolchain select arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu !"
  else
    ARM64_VER=v631
    echo "aarch64 toolchain select gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu !"
  fi
fi

# set toolchain
if [ "${ARM64_VER}" == "v930" ]; then
  CROSS_COMPILE_64=aarch64-linux-
elif [ "${ARM64_VER}" == "v1131" ]; then
  CROSS_COMPILE_64=aarch64-none-linux-gnu-
elif [ "${ARM64_VER}" == "v631" ]; then
  CROSS_COMPILE_64=aarch64-linux-gnu-
fi
CROSS_COMPILE_32=arm-linux-gnueabihf-
CROSS_COMPILE_UCLIBC=arm-cvitek-linux-uclibcgnueabihf-
CROSS_COMPILE_64_NONOS=aarch64-elf-
CROSS_COMPILE_64_NONOS_RISCV64=riscv64-unknown-elf-
CROSS_COMPILE_GLIBC_RISCV64=riscv64-unknown-linux-gnu-
CROSS_COMPILE_MUSL_RISCV64=riscv64-unknown-linux-musl-

# set toolchain path
if [ "${ARM64_VER}" == "v930" ]; then
  CROSS_COMPILE_PATH_64="${HOST_TOOL_PATH}"/gcc/gcc-buildroot-9.3.0-aarch64-linux-gnu
elif [ "${ARM64_VER}" == "v1131" ]; then
  CROSS_COMPILE_PATH_64="${HOST_TOOL_PATH}"/gcc/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu
elif [ "${ARM64_VER}" == "v631" ]; then
  CROSS_COMPILE_PATH_64="${HOST_TOOL_PATH}"/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu
fi
CROSS_COMPILE_PATH_32="${HOST_TOOL_PATH}"/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf
CROSS_COMPILE_PATH_UCLIBC="${HOST_TOOL_PATH}"/gcc/arm-cvitek-linux-uclibcgnueabihf
CROSS_COMPILE_PATH_64_NONOS="${HOST_TOOL_PATH}"/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-elf
CROSS_COMPILE_PATH_64_NONOS_RISCV64="${HOST_TOOL_PATH}"/gcc/riscv64-elf-x86_64
CROSS_COMPILE_PATH_GLIBC_RISCV64="${HOST_TOOL_PATH}"/gcc/riscv64-linux-x86_64
CROSS_COMPILE_PATH_MUSL_RISCV64="${HOST_TOOL_PATH}"/gcc/riscv64-linux-musl-x86_64

# check cross compile
if [ "${SDK_VER}" == 64bit ]; then
  CROSS_COMPILE="$CROSS_COMPILE_64"
  CROSS_COMPILE_PATH="$CROSS_COMPILE_PATH_64"
elif [ "${SDK_VER}" == 32bit ]; then
  CROSS_COMPILE="$CROSS_COMPILE_32"
  CROSS_COMPILE_PATH="$CROSS_COMPILE_PATH_32"
elif [ "${SDK_VER}" == uclibc ]; then
  CROSS_COMPILE="$CROSS_COMPILE_UCLIBC"
  CROSS_COMPILE_PATH="$CROSS_COMPILE_PATH_UCLIBC"
elif [ "${SDK_VER}" == glibc_riscv64 ]; then
  CROSS_COMPILE="$CROSS_COMPILE_GLIBC_RISCV64"
  CROSS_COMPILE_PATH="$CROSS_COMPILE_PATH_GLIBC_RISCV64"
elif [ "${SDK_VER}" == musl_riscv64 ]; then
  CROSS_COMPILE="$CROSS_COMPILE_MUSL_RISCV64"
  CROSS_COMPILE_PATH="$CROSS_COMPILE_PATH_MUSL_RISCV64"
else
  echo -e "Invalid SDK_VER=${SDK_VER}"
  exit 1
fi

# set machine arch
TARGET_MACHINE="$(${CROSS_COMPILE_PATH}/bin/${CROSS_COMPILE}gcc -dumpmachine)"
TOOLCHAIN_FILE="${CVI_TDL_ROOT}"/toolchain/"${TARGET_MACHINE}".cmake

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
else
  echo "Unsupported chip architecture: ${CHIP_ARCH}"
  exit 1
fi

# init build working dir
if [ -d "${BUILD_WORKING_DIR}" ]; then
  echo "Cleanup tmp folder."
  rm -rf ${BUILD_WORKING_DIR}
fi
echo "Creating tmp working directory."
mkdir -p ${BUILD_WORKING_DIR}

# check if cvi_mpi exists
if [ ! -d "$MPI_PATH" ]; then
  echo "Directory $MPI_PATH does not exist. Downloading MPI..."
  if [ "${DUAL_OS}" == "OFF" ]; then
    wget -P ${TOP_DIR} ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/cvi_mpi_rls/cvi_mpi_v410_rls.tar.gz
    tar -xzvf ${TOP_DIR}/cvi_mpi_v410_rls.tar.gz -C "$TOP_DIR"
    rm ${TOP_DIR}/cvi_mpi_v410_rls.tar.gz
    mv ${TOP_DIR}/_temp_cvi_mpi ${MPI_PATH}
    wget -r --no-parent -nH --cut-dirs=8 -P "$MPI_PATH"/lib ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/cvi_mpi_rls/${TARGET_MACHINE}.${CHIP_ARCH,,}/lib
  elif [ "${DUAL_OS}" == "ON" ]; then
    wget -P ${TOP_DIR} ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/cvi_mpi_rls/cvi_mpi_v420_rls.tar.gz
    tar -xzvf ${TOP_DIR}/cvi_mpi_v420_rls.tar.gz -C "$TOP_DIR"
    rm ${TOP_DIR}/cvi_mpi_v420_rls.tar.gz
    wget -r --no-parent -nH --cut-dirs=8 -P "$MPI_PATH"/lib ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/cvi_mpi_rls/${TARGET_MACHINE}.${CHIP_ARCH,,}/lib
  fi
else
    echo "Directory $MPI_PATH exists."
fi
# check if tpu_sdk exists
if [ ! -d "$TPU_SDK_PATH" ]; then
  echo "Directory $TPU_SDK_PATH does not exist. Downloading TPU SDK..."
  if [ "${DUAL_OS}" == "OFF" ]; then
    wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/sdk_release/internal/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvitek_tpu_sdk_internal.tar.gz"
    tar -xzvf ${TOP_DIR}/cvitek_tpu_sdk_internal.tar.gz -C ${TOP_DIR}
    rm ${TOP_DIR}/cvitek_tpu_sdk_internal.tar.gz
  elif [ "${DUAL_OS}" == "ON" ]; then
    wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/sdk_release/internal/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvitek_tpu_sdk_internal.tar.gz"
    tar -xzvf ${TOP_DIR}/cvitek_tpu_sdk_internal.tar.gz -C ${TOP_DIR}
    rm ${TOP_DIR}/cvitek_tpu_sdk_internal.tar.gz
  fi
else
    echo "Directory $TPU_SDK_INSTALL_PATH exists."
fi
# check if ive_sdk exists
if [ "${USE_TPU_IVE}" == "ON" ]; then
  if [ ! -d "$IVE_SDK_PATH" ]; then
    echo "Directory $IVE_SDK_PATH does not exist. Downloading IVE SDK..."
    if [ "${DUAL_OS}" == "OFF" ]; then
      wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/sdk_release/extra/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvitek_ive_sdk.tar.gz"
      tar -xzvf ${TOP_DIR}/cvitek_ive_sdk.tar.gz -C ${TOP_DIR}
      rm ${TOP_DIR}/cvitek_ive_sdk.tar.gz
    elif [ "${DUAL_OS}" == "ON" ]; then
      wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/sdk_release/extra/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvitek_ive_sdk.tar.gz"
      tar -xzvf ${TOP_DIR}/cvitek_ive_sdk.tar.gz -C ${TOP_DIR}
      rm ${TOP_DIR}/cvitek_ive_sdk.tar.gz
    fi
  else
      echo "Directory $IVE_SDK_PATH exists."
  fi
fi
# check if cvi_rtsp exists
if [ ! -d "${TOP_DIR}/cvi_rtsp" ]; then
  echo "Directory ${TOP_DIR}/cvi_rtsp does not exist. Downloading cvi_rtsp..."
  if [ "${DUAL_OS}" == "OFF" ]; then
    wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/sdk_release/sdk/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvi_rtsp.tar.gz"
    mkdir ${TOP_DIR}/cvi_rtsp
    tar -xzvf ${TOP_DIR}/cvi_rtsp.tar.gz -C ${TOP_DIR}/cvi_rtsp
    rm ${TOP_DIR}/cvi_rtsp.tar.gz
  elif [ "${DUAL_OS}" == "ON" ]; then
    wget -P ${TOP_DIR} "ftp://swftp:cvitek@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/sdk_release/sdk/${TARGET_MACHINE}.${CHIP_ARCH,,}/cvi_rtsp.tar.gz"
    mkdir ${TOP_DIR}/cvi_rtsp
    tar -xzvf ${TOP_DIR}/cvi_rtsp.tar.gz -C ${TOP_DIR}/cvi_rtsp
    rm ${TOP_DIR}/cvi_rtsp.tar.gz
  fi
fi

# into tmp/build_sdk
pushd ${BUILD_WORKING_DIR}

# Check cmake version
CMAKE_VERSION="$(cmake --version | grep 'cmake version' | sed 's/cmake version //g')"
CMAKE_REQUIRED_VERSION="3.18.4"
CMAKE_TAR="cmake-3.18.4-Linux-x86_64.tar.gz"
CMAKE_DOWNLOAD_URL="ftp://swftp:cvitek@${FTP_SERVER_IP}/sw_rls/third_party/cmake/${CMAKE_TAR}"
echo "Checking cmake..."
if [ "$(printf '%s\n' "${CMAKE_REQUIRED_VERSION}" "${CMAKE_VERSION}" | sort -V | head -n1)" = "${CMAKE_REQUIRED_VERSION}" ]; then
    CMAKE_BIN=$(command -v cmake)
else
    echo "Cmake version need ${CMAKE_REQUIRED_VERSION}, trying to download from ftp."
    if [ ! -f "${CMAKE_TAR}" ]; then
        wget "${CMAKE_DOWNLOAD_URL}"
    fi
    tar -zxf ${CMAKE_TAR}
    CMAKE_BIN=${PWD}/cmake-3.18.4-Linux-x86_64/bin/cmake
fi
echo "Check cmake done! cmake satisfy!"

# build start
${CMAKE_BIN} -G Ninja ${CVI_TDL_ROOT} -DCVI_PLATFORM=${CHIP_ARCH} \
                                      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
                                      -DENABLE_CVI_TDL_CV_UTILS=ON \
                                      -DMLIR_SDK_ROOT=${TPU_SDK_PATH} \
                                      -DMIDDLEWARE_SDK_ROOT=${MPI_PATH} \
                                      -DTPU_IVE_SDK_ROOT=${IVE_SDK_PATH} \
                                      -DCMAKE_INSTALL_PREFIX=${TDL_SDK_INSTALL_PATH} \
                                      -DTOOLCHAIN_ROOT_DIR=${CROSS_COMPILE_PATH} \
                                      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
                                      -DUSE_TPU_IVE=${USE_TPU_IVE} \
                                      -DBUILD_DOWNLOAD_DIR=${BUILD_DOWNLOAD_DIR} \
                                      -DCONFIG_DUAL_OS=${DUAL_OS} \
                                      -DBUILD_OPTION=${BUILD_OPTION} \
                                      -DTARGET_MACHINE=${TARGET_MACHINE} \
                                      -DMW_VER=${MW_VER} \
                                      -DREPO_USER=$REPO_USER \
                                      -DFTP_SERVER_IP=${FTP_SERVER_IP}

test $? -ne 0 && echo "cmake tdl_sdk failed !!" && popd && exit 1

ninja -j20 || exit 1
ninja install || exit 1
popd
# build end

