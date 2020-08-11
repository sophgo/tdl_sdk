#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVIAI_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$CVIAI_ROOT/tmp

if [[ "$2" == "Asan" ]]; then
    BUILD_TYPE=Asan
else
    BUILD_TYPE=SDKRelease
fi

echo "Creating tmp working directory."
if [[ "$1" == "cmodel" ]]; then
    echo "Temporarily not supported."
    exit
elif [[ "$1" == "soc" ]]; then
    mkdir -p $TMP_WORKING_DIR/build_sdk
    pushd $TMP_WORKING_DIR/build_sdk
    cmake -G Ninja $CVIAI_ROOT -DCVI_TARGET=soc \
                               -DENABLE_SYSTRACE=OFF \
                               -DDISABLE_CLANGTIDY=ON \
                               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                               -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                               -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                               -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                               -DIVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
                               -DTRACER_SDK_ROOT=$TRACER_INSTALL_PATH \
                               -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
                               -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
                               -DCMAKE_TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-aarch64-linux.cmake
    ninja -j8 || exit 1
    ninja install || exit 1
    popd
elif [[ "$1" == "soc32" ]]; then
    mkdir -p $TMP_WORKING_DIR/build_sdk
    pushd $TMP_WORKING_DIR/build_sdk
    cmake -G Ninja $CVIAI_ROOT -DCVI_TARGET=soc \
                               -DENABLE_SYSTRACE=OFF \
                               -DDISABLE_CLANGTIDY=ON \
                               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                               -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                               -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                               -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                               -DIVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
                               -DTRACER_SDK_ROOT=$TRACER_INSTALL_PATH \
                               -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
                               -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
                               -DCMAKE_TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain-gnueabihf-linux.cmake
    ninja -j8 || exit 1
    ninja install || exit 1
    popd
else
  echo "Unsupported build type."
  exit 1
fi
# cd $TMP_WORKING_DIR
# echo "Compressing SDK release..."
# tar cf - ivesdk -P | pv -s $(du -sb ivesdk | awk '{print $1}') | gzip > $CVIAI_ROOT/$IVE_SDK_NAME.tar.gz
# echo "Output md5 sum."
# md5sum $CVIAI_ROOT/$IVE_SDK_NAME.tar.gz
echo "Cleanup tmp folder."
rm -r $TMP_WORKING_DIR
