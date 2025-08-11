# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_cpp)

set(SAMPLE_INCLUDES
    ${IVE_INCLUDES}
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
    ${REPO_DIR}/include/
)

if(BUILD_SHARED)
    set(EX_LIBS tdl_ex)
    set(CORE_LIBS tdl_utils tdl_core ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
else()
    set(CORE_LIBS
        "-Wl,--start-group"
        tdl_utils-static
        tdl_core-static
        ${MIDDLEWARE_LIBS_STATIC}
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        ${IVE_LIBS_STATIC}
        "-Wl,--end-group")
    set(EX_LIBS tdl_ex-static)
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
endif()
if(NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X") AND NOT ("${CVI_PLATFORM}" STREQUAL "CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_blend.cpp)
endif()

if("${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X" OR "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_api_client.cpp)
endif()

if (NOT ("${CVI_PLATFORM}" STREQUAL "CV181X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV184X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_motion_detection.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_img_motion_detection_test.cpp)
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/camera/sample_vi_md.cpp)
endif()

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname})

  set(SAMPLE_LIBS ${CORE_LIBS} dl rt stdc++fs)
  if(name STREQUAL "sample_api_client")
    set(SAMPLE_LIBS ${EX_LIBS} ${CORE_LIBS} dl rt stdc++fs)
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

