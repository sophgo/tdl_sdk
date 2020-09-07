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
    ${CMAKE_SOURCE_DIR}/3rdparty
)

set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
)

add_executable(reg_face_attribute reg_face_attribute.c)
add_executable(reg_face_quality reg_face_quality.c)
add_executable(reg_lfw reg_lfw.c)
add_executable(reg_wider_face reg_wider_face.c)
add_executable(reg_mobiledetv2 reg_mobiledetv2.c)
add_executable(reg_yolov3 reg_yolov3.c)
add_executable(reg_thermal reg_thermal.c)
add_executable(reg_mask_classification reg_mask_classification.c)
add_executable(reg_rgbir_liveness reg_rgbir_liveness.c)
add_executable(reg_mask_fr reg_mask_fr.c)

target_link_libraries(reg_face_attribute cviai_core ${SAMPLE_LIBS})
target_link_libraries(reg_face_quality cviai_core ${SAMPLE_LIBS})
target_link_libraries(reg_lfw cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_wider_face cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_mobiledetv2 cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_yolov3 cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_thermal cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_mask_classification cviai_core ${SAMPLE_LIBS})
target_link_libraries(reg_rgbir_liveness cviai_core cviai_evaluation ${SAMPLE_LIBS})
target_link_libraries(reg_mask_fr cviai_core cviai_evaluation ${SAMPLE_LIBS})
