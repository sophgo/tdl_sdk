#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVIAI_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$CVIAI_ROOT/tmp

if [[ "$1" == "Asan" ]]; then
    BUILD_TYPE=Asan
else
    BUILD_TYPE=SDKRelease
fi

mkdir -p $TMP_WORKING_DIR/build_sdk
pushd $TMP_WORKING_DIR/build_sdk
wget ftp://sa_admin:sa_admin@10.58.65.3//dependency_lib/prebuilt/cmake/cmake-3.18.4-Linux-x86_64.tar.gz -q
tar zxf cmake-3.18.4-Linux-x86_64.tar.gz
CMAKE_BIN=$PWD/cmake-3.18.4-Linux-x86_64/bin/cmake
echo "Creating tmp working directory."

if [[ "$SDK_VER" == "uclibc" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-uclibc-linux.cmake
elif [[ "$SDK_VER" == "64bit" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-aarch64-linux.cmake
elif [[ "$SDK_VER" == "32bit" ]]; then
    TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-gnueabihf-linux.cmake
else
    echo "Wrong SDK_VER=$SDK_VER"
    exit 1
fi

# don't shrink opencv size if platform is 183x series
if [[ "$CHIP_ARCH" != "CV183X" ]]; then
    SHRINK_OPENCV_SIZE=ON
    OPENCV_INSTALL_PATH=""
fi

$CMAKE_BIN -G Ninja $CVIAI_ROOT -DCVI_PLATFORM=$CHIP_ARCH \
                                -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                                -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                                -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                                -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                                -DIVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
                                -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
                                -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
                                -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
                                -DSHRINK_OPENCV_SIZE=$SHRINK_OPENCV_SIZE \
                                -DCVI_MIDDLEWARE_3RD_LDFLAGS="$CVI_TARGET_PACKAGES_LIBDIR" \
                                -DCVI_MIDDLEWARE_3RD_INCCLAGS="$CVI_TARGET_PACKAGES_INCLUDE"

ninja -j8 || exit 1
ninja install || exit 1
popd
echo "Cleanup tmp folder."
rm -rf $TMP_WORKING_DIR

echo "trying to build sample in released folder."
pushd ${AI_SDK_INSTALL_PATH}/sample

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

if [ -z "${CVI_TRACER_ROOT_PATH}" ]; then
    echo "cannot find cvitracer path from environment varables: CVI_TARGET_PACKAGES_LIBDIR and CVI_TARGET_PACKAGES_INCLUDE."
    exit 1
fi

make MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" CVI_TRACER_PATH="$CVI_TRACER_ROOT_PATH" -j10 || exit 1
make MW_PATH="$MW_PATH" TPU_PATH="$TPU_SDK_INSTALL_PATH" IVE_PATH="$IVE_SDK_INSTALL_PATH" CVI_TRACER_PATH="$CVI_TRACER_ROOT_PATH" clean || exit 1
echo "done"
popd