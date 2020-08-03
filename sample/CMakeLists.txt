# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_sample)

function(buildninstall)
  # From https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "SRCS;DEPS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  # note: if it remains unparsed arguments, here, they can be found in variable PARSED_ARGS_UNPARSED_ARGUMENTS
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  message("Provided additional sources are:")
  foreach(src ${PARSED_ARGS_SRCS})
    message("- ${src}")
  endforeach(src)
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
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_jpeg.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
  )
  add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_NAME}.c ${PARSED_ARGS_SRCS})
  target_link_libraries(${PARSED_ARGS_NAME} cviai ${MLIR_LIBS}
                                                  ${OPENCV_LIBS}
                                                  ${MIDDLEWARE_LIBS}
                                                  ${SAMPLE_LIBS})
  install(TARGETS ${PARSED_ARGS_NAME} DESTINATION bin)
  install(FILES ${PARSED_ARGS_NAME}.c DESTINATION sample)
endfunction(buildninstall)

buildninstall(NAME sample_init)
buildninstall(NAME sample_fd SRCS draw_utils.c)
buildninstall(NAME sample_fd_fr SRCS draw_utils.c)
buildninstall(NAME sample_face_quality SRCS draw_utils.c)
buildninstall(NAME sample_object_detection SRCS draw_utils.c)
buildninstall(NAME sample_face_register)
