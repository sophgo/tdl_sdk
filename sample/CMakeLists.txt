# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_sample)

include_directories(
    ${MW_PATH}/include
    ${SDK_PATH}/opencv/include
    ${SDK_PATH}/include
    ${AI_SDK_PATH}/modules/include
    ${TRACER_PATH}
    ${MW_PATH}/sample/common
)

set(SAMPLE_LIBS
    ${SDK_PATH}/lib/libcvikernel.so
    ${SDK_PATH}/lib/libcvimath.so
    ${SDK_PATH}/lib/libcviruntime.so
    ${SDK_PATH}/lib/libcnpy.so
    ${SDK_PATH}/opencv/lib/libopencv_imgcodecs.so
    ${SDK_PATH}/opencv/lib/libopencv_core.so
    ${SDK_PATH}/opencv/lib/libopencv_imgproc.so
    ${MW_PATH}/lib/libsns_full.so
    ${MW_PATH}/lib/libsample.so
    ${MW_PATH}/lib/libvpu.so
    ${MW_PATH}/lib/libisp.so
    ${MW_PATH}/lib/libvenc.so
    ${MW_PATH}/lib/libcvi_vcodec.so
    ${MW_PATH}/lib/libbmjpu.so
    ${MW_PATH}/lib/libawb.so
    ${MW_PATH}/lib/libsys.so
    ${AI_SDK_PATH}/libcvialgo.so
)

add_executable(sample_fd
    sample_fd.c
    draw_utils.c
)

add_executable(sample_object_detection
    sample_object_detection.c
    draw_utils.c
)

target_link_libraries(sample_fd ${SAMPLE_LIBS})
target_link_libraries(sample_object_detection ${SAMPLE_LIBS})

