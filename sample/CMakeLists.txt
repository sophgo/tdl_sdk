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
    ${MIDDLEWARE_SDK_ROOT}/sample/common
    ${BUILD_DOWNLOAD_DIR}/cvi_rtsp/src/cvi_rtsp/include/cvi_rtsp
    ${IVE_INCLUDES}
)

if(NOT ("${CVI_PLATFORM}" STREQUAL "CV181X"  OR "${CVI_PLATFORM}" STREQUAL "CV180X"))
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

if(DEFINED NO_OPENCV)
  buildninstall(NAME sample_init INC ${SAMPLE_INC} DEPS cviai ${SAMPLE_LIBS})
  buildninstall(NAME sample_vi_od INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_intrusion_det INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_md INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  #buildninstall(NAME sample_read_fd INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_aud_cls_read INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_aud_cry INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_read_ped_track INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_read_fdfr INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_vi_fd INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_app_face_capture INC ${SAMPLE_INC} SRCS vi_vo_utils.c DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_face_recog INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai cviai_app ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
else()
  buildninstall(NAME sample_init INC ${SAMPLE_INC} DEPS cviai ${SAMPLE_LIBS})
  buildninstall(NAME sample_aud_esc INC ${SAMPLE_INC} SRCS sample_utils.c DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_aud_soundcmd INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_aud_cls_read INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_aud_cry INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS} ${AUDIO_SAMPLE_LIBS})
  buildninstall(NAME sample_vi_fd INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_fq INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_od INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_intrusion_det INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_obj_counting INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_fall INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_dms INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_fdmask INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_lpdr INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_md INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_object_tracking INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_vi_face_tracking INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c DEPS cviai ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_app_face_capture INC ${SAMPLE_INC} SRCS vi_vo_utils.c DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_app_person_capture INC ${SAMPLE_INC} SRCS vi_vo_utils DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_stream_face_mask_det INC ${SAMPLE_INC} SRCS vi_vo_utils.c DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_stream_face_recognition INC ${SAMPLE_INC} SRCS vi_vo_utils.c DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})
  buildninstall(NAME sample_stream_hardhat INC ${SAMPLE_INC} SRCS vi_vo_utils.c DEPS cviai cviai_app pthread ${SAMPLE_LIBS} ${cvi_rtsp_LIBPATH})

endif()
buildninstall(NAME sample_img_md INC INC ${SAMPLE_INC} DEPS cviai pthread ${SAMPLE_LIBS})
install(FILES vi_vo_utils.c vi_vo_utils.h sample_utils.h sample_utils.c middleware_utils.h sample_log.h middleware_utils.c DESTINATION sample)
install(FILES Makefile DESTINATION sample)
install(DIRECTORY config DESTINATION sample)
