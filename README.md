# Networks

## Preinstall package

### Both

1. cmake
2. Execute ``git submodule update --init --recursive`` to download submodules

### x86_64

1. OpenCV 3.4.0
2. google-glog 3.5.0

**Note:** If you're not using any docker, inside ``scripts`` have installation scripts ``install_glog.sh`` and ``install_opencv.sh``.

### ARM64

1. toolchain files. e.g. ``gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu``
2. BSP release SDK path. ([Download link](https://10.192.22.14), need EasyConnect). e.g. ``soc_bm1880_asic_edb_ai_v1.1.2/soc_bm1880_asic_edb/BM1880_AI_SDK``

Note1: Docker is recommended if your system is **not** Ubuntu 16.04. You'll need to build docker in bmtap2/docker/ubuntu2 first then run the docker patch in ``docker/build.sh``.

Note2: Now you can specify the BSP release SDK as the prebuild package. If it didn't set, will fallback to the prebuild package in the repo.

## Install

You can build code using build script or simply by Cmake commands

### Build script

1. build.sh / build_v2.sh

   build.sh builds with legacy version of bmtap2 (BM1880_DEV) while build_v2.sh build with BM1880_v2. The current branch used is BM1880_v2_RC1.

    ```
        ./build.sh  <board type> <all/soc/cmodel/usb> <build_type> <path to toolchain folder>
                    <path to BSP SDK folder> board type: bm1880
    ```

    for example (bm1880, all release)

    ```
        ./build.sh bm1880 all r <path to toolchain folder> <path to BSP SDK folder>
    ```

    example 2 (bm1880, cmodel release)

    ```
        ./build.sh bm1880 cmodel r
    ```

    example 3 (bm1880, soc release)

    ```
        ./build.sh bm1880 soc r <path to toolchain folder> <path to BSP SDK folder>
    ```

2. scripts/build_bm1880.sh

    ```
        ./build_bm1880.sh  <all/soc/cmodel/usb> <build_type> <path to toolchain folder> <path to BSP SDK folder>
    ```

    for example (all release)

    ```
        ./build.sh all r <path to toolchain folder> <path to BSP SDK folder>
    ```

    example 2 (cmodel release)

    ```
        ./build.sh cmodel r
    ```

    example 3 (soc release)

    ```
        ./build.sh soc r <path to toolchain folder> <path to BSP SDK folder>
    ```

**Note**: For 16.04 users. To build usb mode, you'll have to manually download usb SDK from Github and put the ``include`` and ``lib`` folder under ``prebuilt/usb_bm1880/bmtap2/16.04``. Next, copy the ``bmtap2.h`` from ``prebuilt/cmodel_bm1880/bmtap2/Release/include/`` to ``prebuilt/usb_bm1880/bmtap2/16.04/include/``.

### CMake command

1. bm1880 build cmodel

    ```
        mkdir -p build
        cd build
        mkdir -p cmodel_bm1880
        cd cmodel_bm1880
        cmake ../../ -DCMAKE_BOARD_TYPE=bm1880 -DPLATFORM=cmodel -DUSE_LEGACY_BMTAP2=0 -DCMAKE_INSTALL_PREFIX="../../install"
        make -j8 && make test && make install
    ```

2. bm1880 build soc

    ```
        mkdir -p build
        cd build
        mkdir -p soc_bm1880_asic
        cd soc_bm1880_asic
        cmake ../../ -DCMAKE_BOARD_TYPE=bm1880 -DPLATFORM=soc -DUSE_LEGACY_BMTAP2=0 -DCMAKE_INSTALL_PREFIX="../../install" -DTOOLCHAIN_ROOT_DIR=<path to toolchain folder> -DCMAKE_TOOLCHAIN_FILE="toolchain/toolchain-aarch64-linux.cmake" -DBSPSDK_ROOT_DIR=<path to BSP SDK folder>
        make -j8 && make test && make install
    ```

The following cmake parameters are ON / OFF option switches and their default values.

```
CONFIG_BUILD_SOC     ON
```

### Run Test Codes

#### Cmodel

```
export LD_LIBRARY_PATH=<network dir>/install/cmodel_<board_type>/<build_type>/lib
cd <network dir>/install/cmodel_<board_type>/<build_type>/bin
./test_<net_name> -h
```

For example,

```
export LD_LIBRARY_PATH=<network dir>/install/cmodel_bm1880/Release/lib
cd <network dir>/install/cmodel_bm1880/Release/bin
./test_mtcnn -h
```

#### SoC

Running tests on SoC is similar to Cmodel. You can access a folder in the host PC from SoC with the following command.

```
mount -t nfs <ip>:<mount_folder_dir> /mnt -o nolock
```

#### USB

Running USB mode requires sudo privilege, please combine ``LD_LIBRARY_PATH`` and the execute command together to run.

```
cd <network dir>/install/cmodel_bm1880/Release/bin
sudo LD_LIBRARY_PATH=<network dir>/install/cmodel_bm1880/Release/lib ./test_mtcnn -h
```

### run_check.sh

This script does clang-format, compiling check and clang-tidy in order. To use this tool, you must commit and stash everything before executing the script.

```
./run_check.sh bm1880
```

### Disabling / Enabling GLog

Changing log level is done in runtime. Just add an env variable ``GLOG_minloglevel=(num)``.

```
GLOG_minloglevel=(num)
0 - Debug
1 - Info
2 - Warnings
3 - Errors
```

### CTest

All the unit test codes are in the ``test`` folder. Most of the unit test code accepts validation file with extension *.bmvalid. For example,

```
# No extra space, no extra character!
# if numeric-match is chosen, the order must be the same as the output
# Type: string-match, numeric-match
# Threshold: in the unit of %
# Valid data:
# <data 1>
# <data 2>
Type: numeric-match
Threshold: 5
Valid data:
= 1
```

above is a valid file which check if the output matches 1.

Run the test code with the validation file path at the end of the command, the test code will output ``Test passed`` or ``Test failed``. Note that if type ``numeric-match`` is used, the order of the valid data must be the same as the output.

```
./test_bm_facepose <model dir> <image path> <optional: *.bmvalid file path>
```

To run CTest, navigate to ``build/cmodel_${BM_NETWORKS_TARGET_BASENAME}`` (e.g. build/cmodel_bm1880) and type

```
make test
```

or

```
ctest -V
```

. The former only outputs the tests passed or not, while the latter outputs all the information including the outputs from the test codes.

## OpenCV HW/SW on BM1880

1. Power on bm1880
2. Direct to ``/system/data`` and execute the following commands
```
./load_jpu.sh
insmod bmnpu.ko
mdev -s
```
3. Navigate to ``/home/bitmain`` and execute ``test_imread.sh``
4. The script will compare hardware and software decode output difference.
5. On PC, navigate to ``test/python`` and run ``python imread_python.py`` will get the test cases text files.
```
<py-convert.txt is not available>
py-cvt.txt
py-o.txt
py-resize.txt
py-split.txt
```
6. Copy the following files from SD card to PC and compare them with diff.
```
hw-convert.txt        sw-convert.txt
hw-cvt.txt            sw-cvt.txt
hw-o.txt              sw-o.txt
hw-resize.txt         sw-resize.txt
hw-split.txt          sw-split.txt
```

**Note**:

1. The install prefix is located in ``install`` folder.
2. cmake-gui is recommended to change the settings of CMake.
3. CMAKE_BOARD_TYPE must be set in order to build.
4. If no TOOLCHAIN_ROOT_DIR is set, CMake will fallback to cmodel build.

## Systrace

Systrace python script can be found in Android Studio. You can start a profiling by the following command:

### On PC

```
    python systrace.py --target=linux -o results.html
```

### On SOC

On soc is a bit tricky since bm1880 doesn't have ADB. First copy ``scripts/trace.sh`` to bm1880, then follow the instructions below.

```
cp <mount folder dir>/trace.sh /data/
cd /data
chmod +x trace.sh
./trace.sh <your test program location> <args>
```

When program exits, use the following commands to get the trace logs.

```
cat /sys/kernel/debug/tracing/trace > <mount folder dir>/trace.dat
```

On PC, download [Catapult](https://github.com/catapult-project/catapult/tree/master/) from Github and use the following command to translate trace logs to html.

```
catapult/tracing/bin/trace2html ~/path/to/trace.dat
```

Use Chrome to open the html.

![Trace Image](images/trace.png)

See [systrace](https://developer.android.com/studio/command-line/systrace), [ftrace](https://source.android.com/devices/tech/debug/ftrace) for more info.

## Prebuilt Dependencies

### soc_bm1880_asic

1. bmtap2 bmtap2@88664c0c35ea4ef258ab0ac6da732de52dfbf684
2. bmtap2_v2 bmtap2@dada306d9d67ff381d73bccb1a08a93726946181
3. opencv middleware-soc@e452ae8bb7e62b6cd5a826a6f6f9b3e949c71046
4. ffmpeg middleware-soc@e452ae8bb7e62b6cd5a826a6f6f9b3e949c71046
5. libglog middleware-soc@e452ae8bb7e62b6cd5a826a6f6f9b3e949c71046
6. libprotobuf middleware-soc@666754e5330e010fc2e2dc3e959f1059615f30c46

### usb_bm1880

1. bmtap2 bmtap2@88664c0c35ea4ef258ab0ac6da732de52dfbf684 (Ubuntu 18.10)
2. bmtap2_v2 bmtap2@dada306d9d67ff381d73bccb1a08a93726946181 (modified for compiling)

## Known Issue

1. The result from OpenCV convertTo is different from Python and NEON, all the 0 in convertTo are 1 in Python and NEON. (Currently left untouched.)
2. BMOpenCV pads cv::Mat if it's not the multiplier of 16x if width exceed a length. (Length not known yet.) This can be handled with the ``int cv::Mat::step`` parameter in cv::Mat.
3. Hardware decode in bm1880 is not correct. Use software decode for now.
4. In USB mode, some bmodels are not working proerly. e.g. tinyssh 90, ssh 160, ssh 90.

Solved:
1. A bug is found in bmtap2 issue[#1](#1) cause program crash if reusing a command buffer.