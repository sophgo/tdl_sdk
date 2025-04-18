# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_cpp)


set(SAMPLE_INCLUDES
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
)

set(SAMPLE_LIBS ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
include_directories(${SAMPLE_INCLUDES})


file(GLOB SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(("${CVI_PLATFORM}" STREQUAL "CV180X") OR 
   ("${CVI_PLATFORM}" STREQUAL "CV181X") OR 
   ("${CVI_PLATFORM}" STREQUAL "CV182X") OR
   ("${CVI_PLATFORM}" STREQUAL "CV183X") OR
   ("${CVI_PLATFORM}" STREQUAL "SOPHON") OR
   ("${CVI_PLATFORM}" STREQUAL "CMODEL_CVITEK"))
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_qwenvl_process.cpp)
endif()

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname})
  target_link_libraries(${name} cvi_tdl ${SAMPLE_LIBS} stdc++fs)
  install(TARGETS ${name} DESTINATION bin)
endforeach ()

