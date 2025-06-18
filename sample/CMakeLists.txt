# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)

project(sample)

function(buildninstall)
  # From https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "INC;SRCS;DEPS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  # note: if it remains unparsed arguments, here, they can be found in variable PARSED_ARGS_UNPARSED_ARGUMENTS
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  project(${PARSED_ARGS_NAME})
  # For debug purpose.
  # message("Provided additional sources are:")
  # foreach(src ${PARSED_ARGS_SRCS})
  #   message("- ${src}")
  # endforeach(src)
  include_directories(${PARSED_ARGS_INC})
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.c ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS} atomic)
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
  string(SUBSTRING "${PROJECT_NAME}" 0 10 substring1)
  if (substring1 STREQUAL "sample_app")
      install(FILES ${PROJECT_NAME}.c DESTINATION sample/cvi_tdl_app/)
  else ()
      install(FILES ${PROJECT_NAME}.c DESTINATION sample/cvi_tdl/)
  endif()
endfunction(buildninstall)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_app 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/lib/cvi_kit)
set(REG_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${MIDDLEWARE_SDK_ROOT}/component/isp/common
)

if(${CVI_PLATFORM} STREQUAL "SOPHON")
    set(SAMPLE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
        ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so
    )
elseif(${CONFIG_DUAL_OS} STREQUAL "ON")
    set(SAMPLE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libmsg.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvilink.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MLIR_SDK_ROOT}/lib/libz.so
    )
else()
    set(SAMPLE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MLIR_SDK_ROOT}/lib/libz.so
    )

    if("${MW_VER}" STREQUAL "v2")
        if(${CVI_PLATFORM} STRLESS "SOPHON")
            set(SAMPLE_LIBS
                ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
                ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
                ${MLIR_SDK_ROOT}/lib/libz.so
            )
        endif()
        set(SAMPLE_LIBS ${SAMPLE_LIBS}
            ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
        )
    else()
        set(SAMPLE_LIBS ${SAMPLE_LIBS}
            ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
        )
    endif()

    if (NOT "${CVI_PLATFORM}" STREQUAL "CV183X")
        set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
    endif()
endif()

set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MLIR_LIBS} ${OPENCV_LIBS} ${IVE_LIBS})

if (NOT DEFINED NO_OPENCV)
    buildninstall(NAME sample_img_face_recognition INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_kit ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_lane_det INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_kit ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_pose INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_kit ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_lpd_lpr INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_lpd_lpr_keypoint INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_yolov8_seg INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_fatigue_eye INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_fatigue_yawn INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_face_quality INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_face_liveness INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
    buildninstall(NAME sample_img_occlusion_cls INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS})
endif()

buildninstall(NAME sample_img_detection_roi INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_img_face_landmarker INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
buildninstall(NAME sample_img_face_mask INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
buildninstall(NAME sample_img_face_attribute_cls INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
buildninstall(NAME sample_img_face_det INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_img_hand_det INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_img_hand_cls INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_img_hand_keypoint INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_img_topformer_seg INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
buildninstall(NAME sample_audio_cls INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/sys_utils.c)
buildninstall(NAME sample_blip_itm INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/sys_utils.c)
buildninstall(NAME sample_blip_vqa INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/sys_utils.c)
buildninstall(NAME sample_blip_cap INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/sys_utils.c)
buildninstall(NAME sample_yolo_world_v2 INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app cvi_kit ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/sys_utils.c)
buildninstall(NAME sample_img_det INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_tdl_app ${SAMPLE_LIBS})
buildninstall(NAME sample_img_face_feature_extract INC ${REG_INCLUDES} DEPS cvi_tdl atomic cvi_kit ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov8_roi INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})

# yolo external sample
buildninstall(NAME sample_yolov5 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov6 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov7 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov8 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov10 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_yolov11 INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS} cvi_kit)
buildninstall(NAME sample_yolox INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
buildninstall(NAME sample_ppyoloe INC ${REG_INCLUDES} DEPS cvi_tdl atomic ${SAMPLE_LIBS})
