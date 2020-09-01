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
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
  install(FILES ${PROJECT_NAME}.c DESTINATION sample)
endfunction(buildninstall)

set(SAMPLE_INC
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/opencv/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${TRACER_PATH}
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${IVE_INCLUDES})

set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_jpeg.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so)

buildninstall(NAME sample_init INC ${SAMPLE_INC} DEPS cviai_core)
buildninstall(NAME sample_read_image INC ${SAMPLE_INC} DEPS cviai_core ${IVE_LIBS})
buildninstall(NAME sample_fd INC ${SAMPLE_INC} DEPS cviai_core cviai_frservice ${SAMPLE_LIBS})
buildninstall(NAME sample_face_quality INC ${SAMPLE_INC} DEPS cviai_core cviai_frservice ${SAMPLE_LIBS})
buildninstall(NAME sample_object_detection INC ${SAMPLE_INC} DEPS cviai_core cviai_objservice ${SAMPLE_LIBS})
buildninstall(NAME sample_face_register INC ${SAMPLE_INC} DEPS cviai_core ${SAMPLE_LIBS})
buildninstall(NAME sample_mask_face_recognition INC ${SAMPLE_INC} DEPS cviai_core cviai_frservice ${SAMPLE_LIBS})

buildninstall(NAME sample_digital_tracking INC ${SAMPLE_INC} DEPS cviai_core cviai_frservice ${IVE_LIBS})
buildninstall(NAME sample_tamper_detection INC ${SAMPLE_INC} DEPS cviai_core ${IVE_LIBS})
