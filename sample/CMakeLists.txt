# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_sample)

include_directories(
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/opencv/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${TRACER_PATH}
    ${MIDDLEWARE_SDK_ROOT}/sample/common
)

set(SAMPLE_LIBS
    ${MLIR_SDK_ROOT}/lib/libcvikernel.so
    ${MLIR_SDK_ROOT}/lib/libcvimath.so
    ${MLIR_SDK_ROOT}/lib/libcviruntime.so
    ${MLIR_SDK_ROOT}/lib/libcnpy.so
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_imgcodecs.so
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_core.so
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_imgproc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libbmjpu.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
)

#add_executable(sample_fd
#    sample_fd.c
#    draw_utils.c
#)

add_executable(sample_object_detection
    sample_object_detection.c
    draw_utils.c
)

#target_link_libraries(sample_fd ${SAMPLE_LIBS})
target_link_libraries(sample_object_detection cviai ${MLIR_LIBS} ${OPENCV_LIBS} ${MIDDLEWARE_LIBS} ${SAMPLE_LIBS})

