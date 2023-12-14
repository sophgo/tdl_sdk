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
  #install(TARGETS ${PROJECT_NAME} DESTINATION regression)
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
  if (PROJECT_NAME STREQUAL "test_main")
      install(TARGETS ${PROJECT_NAME} DESTINATION regression)
  endif ()
endfunction(buildninstallcpp)

set(REG_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/opencv/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_app
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_evaluation
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${CVI_RTSP_INCLUDE}
)

if(${CVI_PLATFORM} STREQUAL "CV186X")
set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
    ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so
    ${CVI_RTSP_LIBPATH})
elseif(${CONFIG_DUAL_OS} STREQUAL "ON")
set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libmsg.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvpu.so
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
    ${CVI_RTSP_LIBPATH}
)


if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "CV181X")
  list(APPEND SAMPLE_LIBS ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV180X")
  list(APPEND SAMPLE_LIBS ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so)
else()
  list(APPEND SAMPLE_LIBS ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so)
endif()
endif()


if(${CVI_PLATFORM} STREQUAL "CV186X")
  set(UnitTests
  reg_daily_petdet.cpp
  reg_daily_handdet.cpp
  reg_daily_meet.cpp
  reg_daily_person_vehicle.cpp
  reg_daily_hardhat_det.cpp
  reg_daily_ir_scrfd_face.cpp
  reg_daily_scrfd_det.cpp
  reg_daily_lp_det.cpp
  )
else()
set(UnitTests
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
  reg_daily_fdmask.cpp
  reg_daily_es_classification.cpp
  reg_daily_eye_classification.cpp
  reg_daily_yawn_classification.cpp
  reg_daily_fl.cpp
  reg_daily_soundcmd.cpp
  reg_daily_incarod.cpp
  reg_daily_face_cap.cpp
  reg_daily_feature_matching.cpp
)
endif()

set(UnitTest_SRCS
  cvi_tdl_test.cpp
  raii.cpp
  regression_utils.cpp
  ${UnitTests}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../modules/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/log
)

buildninstallcpp(NAME test_main INC ${REG_INCLUDES} SRCS ${UnitTest_SRCS} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app gtest_main ${SAMPLE_LIBS})

if ("${CVI_PLATFORM}" STREQUAL "CV186X")
  set(DAILY_REGRESSION_ASSETS "assets_186x")
elseif("${CVI_PLATFORM}" STREQUAL "CV183X")
  set(DAILY_REGRESSION_ASSETS "assets_183x")
elseif("${CVI_PLATFORM}" STREQUAL "CV182X" )
  set(DAILY_REGRESSION_ASSETS "assets_182x")
elseif("${CVI_PLATFORM}" STREQUAL "CV181X" )
  set(DAILY_REGRESSION_ASSETS "assets_181x")
elseif("${CVI_PLATFORM}" STREQUAL "CV180X" )
  set(DAILY_REGRESSION_ASSETS "assets_180x")
endif()

file(GLOB asset_files "${DAILY_REGRESSION_ASSETS}/*.json")
install(DIRECTORY DESTINATION ${assets})

foreach(file ${asset_files})
  get_filename_component(asset_realpath "${file}" REALPATH)
  install(FILES ${asset_realpath} DESTINATION regression/assets)
endforeach()

install(FILES daily_regression.sh PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE DESTINATION regression)
