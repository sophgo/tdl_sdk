# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_cpp)

set(SAMPLE_INCLUDES
    ${IVE_INCLUDES}
    ${MLIR_INCLUDES}
    ${FBANK_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
    ${REPO_DIR}/include/
    ${REPO_DIR}/src/components/nn
)

if(BUILD_SHARED)
    set(EX_LIBS tdl_ex ${LIBWEBSOCKETS_LIBS}
    ${OPENSSL_LIBRARY}
    ${ZLIB_LIBRARY})
    set(CORE_LIBS tdl_utils tdl_core ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES} ${MIDDLEWARE_OBJ})
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--start-group")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--whole-archive")

    set(CORE_LIBS
        tdl_core-static
        tdl_utils-static
        ${MIDDLEWARE_LIBS_STATIC}
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        ${IVE_LIBS_STATIC}
    )
    set(EX_LIBS tdl_ex-static)

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-whole-archive")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--end-group")
endif()
include_directories(${SAMPLE_INCLUDES})


file(GLOB_RECURSE SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(("${CVI_PLATFORM}" STREQUAL "CV180X") OR
   ("${CVI_PLATFORM}" STREQUAL "CV181X") OR
   ("${CVI_PLATFORM}" STREQUAL "CV182X") OR
   ("${CVI_PLATFORM}" STREQUAL "CV183X") OR
   ("${CVI_PLATFORM}" STREQUAL "CV184X") OR
   ("${CVI_PLATFORM}" STREQUAL "SOPHON") OR
   ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X") OR
   ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_qwenvl_process.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_vidmulti.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_clip_mp4.cpp)
endif()
if(NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X") AND NOT ("${CVI_PLATFORM}" STREQUAL "CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_blend.cpp)
endif()

if(("${CVI_PLATFORM}" STREQUAL "CV180X") OR
  ("${CVI_PLATFORM}" STREQUAL "CV181X") OR
  ("${CVI_PLATFORM}" STREQUAL "CV182X") OR
  ("${CVI_PLATFORM}" STREQUAL "CV183X") OR
  ("${CVI_PLATFORM}" STREQUAL "CV184X"))
  if(BUILD_SHARED)
    set(EX_LIBS ${EX_LIBS} tdl_utils)
  endif()
endif()

if (NOT ("${CVI_PLATFORM}" STREQUAL "CV181X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV184X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_motion_detection.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_motion_detection_test.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/camera/sample_vi_md.cpp)
endif()

if ("${CVI_PLATFORM}" STREQUAL "SOPHON" OR
    "${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X" OR
    "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_api_client.cpp)
endif()


if (NOT ("${CVI_PLATFORM}" STREQUAL "CV180X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV181X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV182X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV183X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_microphone_audio_asr.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_microphone_audio_vad_asr.cpp)
endif()

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)

  string(FIND "${name}" "_asr" pos)
  if(pos GREATER_EQUAL 0 AND DEFINED ENABLE_SPEECH_RECOGNITION AND NOT ENABLE_SPEECH_RECOGNITION)
    continue()
  endif()

  string(FIND "${name}" "_app" pos)
  if(pos GREATER_EQUAL 0 AND DEFINED ENABLE_APP_PIPELINE AND NOT ENABLE_APP_PIPELINE)
    continue()
  endif()


  add_executable(${name} ${fname})

  set(SAMPLE_LIBS ${CORE_LIBS} dl rt stdc++fs)
  if(name STREQUAL "sample_api_client")
    set(SAMPLE_LIBS ${EX_LIBS} ${CORE_LIBS} dl rt stdc++fs)
  elseif(name STREQUAL "sample_vi_md")
    set(SAMPLE_LIBS ${EX_LIBS} ${CORE_LIBS})
  elseif(name MATCHES "microphone_audio")
    if(BUILD_SHARED)
      set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_AUDIO_LIBS})
    else()
      set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_AUDIO_LIBS_STATIC})
    endif()
  endif()

  target_link_libraries(${name} ${SAMPLE_LIBS})
  #strip
  if(NOT BUILD_SHARED)
    add_custom_command(TARGET ${name} POST_BUILD
          COMMAND ${CROSS_COMPILE}strip $<TARGET_FILE:${name}>
          COMMENT "Stripping binary for ${name}")
  endif()
  install(TARGETS ${name} DESTINATION bin/cpp)
endforeach ()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config DESTINATION .)
