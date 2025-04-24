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
    set(SAMPLE_LIBS cvi_tdl ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
else()
    set(SAMPLE_LIBS
        "-Wl,--start-group"
        ${cvi_tdl_BINARY_DIR}/libcvi_tdl-static.a
        ${MIDDLEWARE_LIBS_STATIC}
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        "-Wl,--end-group")
endif()

include_directories(${SAMPLE_INCLUDES})

file(GLOB SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
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
  install(TARGETS ${name} DESTINATION bin)
endforeach ()
