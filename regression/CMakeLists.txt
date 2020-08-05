# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_regression)

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
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    #${MIDDLEWARE_SDK_ROOT}/lib/libbmjpu.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
)

add_executable(reg_face_attribute reg_face_attribute.c)
add_executable(reg_face_quality reg_face_quality.c)

target_link_libraries(reg_face_attribute cviai_core ${TRACER_LIBS} ${MLIR_LIBS} ${OPENCV_LIBS} ${MIDDLEWARE_LIBS} ${SAMPLE_LIBS})
target_link_libraries(reg_face_quality cviai_core ${TRACER_LIBS} ${MLIR_LIBS} ${OPENCV_LIBS} ${MIDDLEWARE_LIBS} ${SAMPLE_LIBS})

