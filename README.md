# CVIAI

## How to build

### SoC mode

1. 64-bit

```
$ mkdir build
$ cd build
$ cmake -G Ninja .. -DCVI_TARGET=soc \
                    -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-aarch64-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DTRACER_SDK_ROOT=<tracer_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja && ninja install
```

1. 32-bit

```
$ mkdir build_soc
$ cd build
$ cmake -G Ninja .. -DCVI_TARGET=soc \
                    -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-gnueabihf-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DTRACER_SDK_ROOT=<tracer_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8
```