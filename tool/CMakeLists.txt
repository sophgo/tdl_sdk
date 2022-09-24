# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_tools)

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
  install(TARGETS ${PROJECT_NAME} DESTINATION bin/tool)
  # install(FILES ${PROJECT_NAME}.c DESTINATION tool)
endfunction(buildninstall)

function(buildninstallcpp)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "INC;SRCS;DEPS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  project(${PARSED_ARGS_NAME})
  include_directories(${PARSED_ARGS_INC})
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})
  install(TARGETS ${PROJECT_NAME} DESTINATION bin/tool)
  # install(FILES ${PROJECT_NAME}.cpp DESTINATION tool)
endfunction(buildninstallcpp)

set(SAMPLE_INC
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/opencv/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${IVE_INCLUDES}
    ${BUILD_DOWNLOAD_DIR}/cvi_rtsp/src/cvi_rtsp/include/cvi_rtsp
)

if(NOT "${CVI_PLATFORM}" STREQUAL "MARS")
set(SAMPLE_INC ${SAMPLE_INC}
    ${TRACER_PATH}
)
endif()

set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MLIR_SDK_ROOT}/lib/libz.so
    ${BUILD_DOWNLOAD_DIR}/cvi_rtsp/src/cvi_rtsp/src/libcvi_rtsp.so
)

if("${CVI_PLATFORM}" STREQUAL "MARS" OR "${CVI_PLATFORM}" STREQUAL "PHOBOS")
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
)
else()
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
)
endif()


if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "MARS")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

buildninstall(NAME sample_read_mot INC ${SAMPLE_INC} SRCS utils/od.c sample_mot_utils.c DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME visual_app_face_capture INC ${SAMPLE_INC}  SRCS vi_vo_utils.c DEPS cviai cviai_app ${SAMPLE_LIBS})

buildninstallcpp(NAME mot_dump_data INC ${SAMPLE_INC}
                 SRCS utils/od.c utils/mot_base.cpp utils/mot_evaluation.cpp 
                 DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME mot_auto_tuning INC ${SAMPLE_INC}
                 SRCS utils/od.c utils/mot_base.cpp utils/mot_evaluation.cpp utils/mot_auto_tuning_helper.cpp
                 DEPS cviai ${SAMPLE_LIBS})

install(DIRECTORY utils DESTINATION tool)
