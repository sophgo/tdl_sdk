# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_c)

set(SAMPLE_INCLUDES
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
    ${REPO_DIR}/include/c_apis
    ${REPO_DIR}/sample/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

if(BUILD_SHARED)
    set(SAMPLE_LIBS tdl ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
else()
    set(SAMPLE_LIBS
        "-Wl,--start-group"
        ${CMAKE_INSTALL_PREFIX}/lib/libtdl-static.a
        "-Wl,--whole-archive"
        ${MIDDLEWARE_LIBS_STATIC}
        "-Wl,--no-whole-archive"
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        ${IVE_LIBS_STATIC}
        "-Wl,--end-group"
)
endif()

include_directories(${SAMPLE_INCLUDES})

file(GLOB SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
if(${CVI_PLATFORM} STREQUAL "CV180X" OR
   ${CVI_PLATFORM} STREQUAL "CV181X" OR
   ${CVI_PLATFORM} STREQUAL "CV182X" OR
   ${CVI_PLATFORM} STREQUAL "CV183X" OR
   ${CVI_PLATFORM} STREQUAL "CV184X" OR
   ${CVI_PLATFORM} STREQUAL "SOPHON")
  file(GLOB CAMERA_SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/camera/*.c)
  list(APPEND SAMPLE_SRC_FILES ${CAMERA_SAMPLE_SRC_FILES})
endif()

if(NOT ${CVI_PLATFORM} STREQUAL "CV181X" AND NOT ${CVI_PLATFORM} STREQUAL "CV184X")
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_motion_detection.c)
endif()

file(GLOB UTILS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname} ${UTILS_CPP})
  target_link_libraries(${name} ${SAMPLE_LIBS} dl rt atomic m)
  #strip
  if(NOT BUILD_SHARED)
    add_custom_command(TARGET ${name} POST_BUILD
          COMMAND ${CROSS_COMPILE}strip $<TARGET_FILE:${name}>
          COMMENT "Stripping binary for ${name}")
  endif()
  install(TARGETS ${name} DESTINATION bin/c)
endforeach ()
