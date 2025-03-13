# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_c)


set(SAMPLE_INCLUDES
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include/tdl_sdk/framework
    ${REPO_DIR}/include/tdl_sdk/components
    ${REPO_DIR}/include/tdl_sdk/app
    ${REPO_DIR}/include/tdl_sdk/nn
    ${REPO_DIR}/include/tdl_sdk/c_apis
    ${REPO_DIR}/include
)

set(SAMPLE_LIBS ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
include_directories(${SAMPLE_INCLUDES})


file(GLOB SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname})
  target_link_libraries(${name} cvi_tdl ${SAMPLE_LIBS} atomic)
  install(TARGETS ${name} DESTINATION bin)
endforeach ()
