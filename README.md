# CVIAI

Programming guide is available under ``doc/``.

## How to build

### SoC mode

Replace the ``<XXXXXX>`` with the path on your PC.

1. 64-bit

```
$ mkdir build
$ cd build
$ cmake -G Ninja .. -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-aarch64-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DOPENCV_ROOT=<opencv_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8 && ninja install
```

2. 32-bit

```
$ mkdir build_soc
$ cd build
$ cmake -G Ninja .. -DTOOLCHAIN_ROOT_DIR=<toolchain_root_dir> \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-gnueabihf-linux.cmake \
                    -DMLIR_SDK_ROOT=<mlir_root_dir> \
                    -DOPENCV_ROOT=<opencv_root_dir> \
                    -DIVE_SDK_ROOT=<ive_root_dir> \
                    -DMIDDLEWARE_SDK_ROOT=<middleware_root_dir> \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8 && ninja install
```

**Note**

1. ``OPENCV_ROOT`` may be inside ``<mlir_root_dir>/opencv``.
2. ``mmf.tar.gz`` contains all the required libraries, use ``mw.tar.gz`` instead.
3. Perfetto only supports GCC version >= 7. Please update your local toolchain to meet the requirement.

## Coding Rules

1. Files in ``include`` should be C style. C++ style stuffs should stay in ``modules``.
2. Please put the AI inference function in correct category. In same category, put the function in alphabetical order.
