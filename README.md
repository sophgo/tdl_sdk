# CVIAI

## How to build

### SoC mode

Replace the ``<XXXXXX>`` with the path on your PC.

1. 64-bit

```
$ mkdir build
$ cd build
$ cmake -G Ninja .. -DCVI_TARGET=soc \
                    -DENABLE_SYSTRACE=ON \
                    -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-aarch64-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DOPENCV_ROOT=<opencv_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja && ninja install
```

1. 32-bit

```
$ mkdir build_soc
$ cd build
$ cmake -G Ninja .. -DCVI_TARGET=soc \
                    -DENABLE_SYSTRACE=ON \
                    -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-gnueabihf-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DOPENCV_ROOT=<opencv_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8
```

**Note**:

1. ``OPENCV_ROOT`` may be inside ``<mlir_root_dir>/opencv``
2. ``mmf.tar.gz`` contains all the required libraries.

