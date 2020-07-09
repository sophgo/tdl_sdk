
# Build command

mkdir build
cd build
cmake .. -DMW_PATH=/home/terry/下載 -DSDK_PATH=/home/terry/下載/cvitek_tpu_sdk -DTRACER_PATH=/home/terry/work/tracer -DAI_SDK_PATH=/home/terry/work/cviai -DTOOLCHAIN_ROOT_DIR=/usr/local/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/ -DCMAKE_TOOLCHAIN_FILE=${PWD}/../../toolchain/toolchain-aarch64-linux.cmake
make
