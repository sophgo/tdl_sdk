# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_regression)

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
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})
  install(TARGETS ${PROJECT_NAME} DESTINATION regression)
endfunction(buildninstall)

function(buildninstallcpp)
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
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})
  install(TARGETS ${PROJECT_NAME} DESTINATION regression)
endfunction(buildninstallcpp)

set(REG_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/opencv/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${TRACER_PATH}
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${CMAKE_SOURCE_DIR}/3rdparty
)

set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
)

if("${CVI_PLATFORM}" STREQUAL "CV182X")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

buildninstall(NAME reg_face_attribute INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_recognition INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_quality INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_lfw INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_wider_face INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mobiledetv2 INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_yolov3 INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_thermal INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mask_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_rgbir_liveness INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mask_fr INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_object_intersect INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_multi_object_tracking INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_reid INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_align INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_es_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_deeplabv3 INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_lpdr INC ${REG_INCLUDES} DEPS cviai_core cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_feature_matching INC ${REG_INCLUDES} DEPS cviai_core DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_fall_detection INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_landmark INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_eye_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_yawn_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_incar_od INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_service cviai_evaluation ${SAMPLE_LIBS})

buildninstallcpp(NAME reg_multithread INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_fq INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_thermal_fd INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_reid INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_td INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_lpd INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_lpr INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_mot INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_mask_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_fr INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_fl INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_yawn_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_eye_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_es_classification INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME reg_daily_incarod INC ${REG_INCLUDES} DEPS cviai_core cviai_service cviai_evaluation ${SAMPLE_LIBS})

install(DIRECTORY assets DESTINATION regression)
