

project(test)

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
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
endfunction(buildninstallcpp)

set(REG_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MLIR_SDK_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/core
    ${TRACER_PATH}
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${CMAKE_SOURCE_DIR}/3rdparty
    ${BUILD_DOWNLOAD_DIR}/cvi_rtsp/src/cvi_rtsp/include/cvi_rtsp
)


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
)

set(SAMPLE_LIBS ${SAMPLE_LIBS} ${BUILD_DOWNLOAD_DIR}/cvi_rtsp/src/cvi_rtsp/src/libcvi_rtsp.so)

set(AUDIO_SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libcli.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_ssp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_audio.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vqe.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libtinyalsa.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_VoiceEngine.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_RES1.so)

if("${CVI_PLATFORM}" STREQUAL "CV181X"  OR "${CVI_PLATFORM}" STREQUAL "CV180X")
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MLIR_SDK_ROOT}/lib/libz.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
)
else()
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
)
endif()


if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV180X")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

buildninstallcpp(NAME test_face_cap INC ${SAMPLE_INC} DEPS cviai  cviai_app ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_fd INC ${SAMPLE_INC} DEPS cviai  cviai_app ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
# buildninstallcpp(NAME cviai_run_face_cap INC ${SAMPLE_INC} DEPS cviai_core cviai_service cviai_app ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
set(SAMPLE_LIBS_SIMPLE
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
)
if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV180X")
  set(SAMPLE_LIBS_SIMPLE ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()
buildninstallcpp(NAME test_img_cvimd INC ${MIDDLEWARE_SDK_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib ${IVE_INCLUDES} DEPS cvi_md ${SAMPLE_LIBS_SIMPLE} ${IVE_LIBS} ${MIDDLEWARE_LIBS})
