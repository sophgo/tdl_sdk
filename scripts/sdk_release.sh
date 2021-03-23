#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CVIAI_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$CVIAI_ROOT/tmp

if [[ "$2" == "Asan" ]]; then
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
$CMAKE_BIN -G Ninja $CVIAI_ROOT -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
                                -DOPENCV_ROOT=$OPENCV_INSTALL_PATH \
                                -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                                -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                                -DIVE_SDK_ROOT=$IVE_SDK_INSTALL_PATH \
                                -DCMAKE_INSTALL_PREFIX=$AI_SDK_INSTALL_PATH \
                                -DCMAKE_TOOLCHAIN_FILE=$CVIAI_ROOT/toolchain/toolchain.cmake
ninja -j8 || exit 1
ninja install || exit 1
popd
echo "Cleanup tmp folder."
rm -r $TMP_WORKING_DIR
