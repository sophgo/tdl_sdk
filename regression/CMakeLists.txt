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
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MLIR_SDK_ROOT}/lib/libz.so
)

set(UnitTests
  reg_daily_feature_matching.cpp
  reg_daily_fq.cpp
  reg_daily_mobiledet.cpp
  reg_daily_md.cpp
  reg_daily_fr.cpp
  reg_daily_mask_classification.cpp
  reg_daily_lpd.cpp
  reg_daily_lpr.cpp
  reg_daily_mot.cpp
  reg_daily_reid.cpp
  reg_daily_thermal_fd.cpp
  reg_daily_thermal_person_detection.cpp
  reg_daily_core.cpp
  reg_daily_liveness.cpp
  reg_daily_alphapose.cpp
  reg_daily_fall.cpp
  reg_daily_retinaface.cpp
  reg_daily_retinafaceIR.cpp
  reg_daily_retinaface_hardhat.cpp
  reg_daily_es_classification.cpp
  reg_daily_eye_classification.cpp
  reg_daily_yawn_classification.cpp
  reg_daily_fl.cpp
  reg_daily_soundcmd.cpp
  reg_daily_incarod.cpp
  reg_daily_td.cpp
)

set(UnitTest_SRCS
  cviai_test.cpp
  raii.cpp
  regression_utils.cpp
  ${UnitTests}
)

if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "MARS")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

if("${CVI_PLATFORM}" STREQUAL "MARS")
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
)
else()
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../modules/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/log
)

buildninstall(NAME reg_face_attribute INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_recognition INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_quality INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_lfw INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_wider_face INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mobiledetv2 INC ${REG_INCLUDES} SRCS od_utils.c DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_yolov3 INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_thermal INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mask_classification INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_rgbir_liveness INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_mask_fr INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_multi_object_tracking INC ${REG_INCLUDES} SRCS od_utils.c DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_reid INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_align INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_es_classification INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_lpdr INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_face_landmark INC ${REG_INCLUDES} DEPS cviai   cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_eye_classification INC ${REG_INCLUDES} DEPS cviai   cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_yawn_classification INC ${REG_INCLUDES} DEPS cviai   cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_incar_od INC ${REG_INCLUDES} DEPS cviai   cviai_evaluation ${SAMPLE_LIBS})

buildninstallcpp(NAME reg_multithread INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_main INC ${REG_INCLUDES} SRCS ${UnitTest_SRCS} DEPS cviai  cviai_evaluation gtest_main ${SAMPLE_LIBS})
buildninstall(NAME reg_soundcmd INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})

buildninstall(NAME reg_deeplabv3 INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstall(NAME reg_fall_detection INC ${REG_INCLUDES} DEPS cviai   cviai_evaluation ${SAMPLE_LIBS})

if ("${CVI_PLATFORM}" STREQUAL "CV183X")
  set(DAILY_REGRESSION_ASSETS "assets_183x")
elseif("${CVI_PLATFORM}" STREQUAL "MARS")
  set(DAILY_REGRESSION_ASSETS "assets_mars")
else()
  set(DAILY_REGRESSION_ASSETS "assets_182x")
endif()

file(GLOB asset_files "${DAILY_REGRESSION_ASSETS}/*.json")
install(DIRECTORY DESTINATION ${assets})

foreach(file ${asset_files})
  get_filename_component(asset_realpath "${file}" REALPATH)
  install(FILES ${asset_realpath} DESTINATION regression/assets)
endforeach()

install(FILES daily_regression.sh PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE DESTINATION regression)
