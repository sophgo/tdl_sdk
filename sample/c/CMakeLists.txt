# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(sample_c)

set(SAMPLE_INCLUDES
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${REPO_DIR}/include
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
    ${REPO_DIR}/include/c_apis
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Add subdirectories
add_subdirectory(utils)
if (${CVI_PLATFORM} STREQUAL "CV181X" OR ${CVI_PLATFORM} STREQUAL "CV184X")
    add_subdirectory(camera)
endif()

if(BUILD_SHARED)
    set(SAMPLE_LIBS tdl tdl_utils ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
else()
    set(SAMPLE_STATIC_LIBS
        tdl-static
        tdl_utils-static
        ${MIDDLEWARE_LIBS_STATIC}
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        ${IVE_LIBS_STATIC}
    )
    
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--start-group")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--whole-archive")
    set(SAMPLE_LIBS ${SAMPLE_STATIC_LIBS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-whole-archive")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--end-group")
endif()

include_directories(${SAMPLE_INCLUDES})

file(GLOB SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
if(${CVI_PLATFORM} STREQUAL "CV180X" OR
   ${CVI_PLATFORM} STREQUAL "CV181X" OR
   ${CVI_PLATFORM} STREQUAL "CV182X" OR
   ${CVI_PLATFORM} STREQUAL "CV183X" OR
   ${CVI_PLATFORM} STREQUAL "CV184X" OR
   ${CVI_PLATFORM} STREQUAL "SOPHON")
  file(GLOB APP_SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/app/*.c)
  list(APPEND SAMPLE_SRC_FILES ${APP_SAMPLE_SRC_FILES})
endif()

if(NOT ${CVI_PLATFORM} STREQUAL "CV181X" AND NOT ${CVI_PLATFORM} STREQUAL "CV184X")
  list(REMOVE_ITEM SAMPLE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_motion_detection.c)
endif()

message(STATUS "SAMPLE_SRC_FILES:${SAMPLE_SRC_FILES}")

foreach (fname ${SAMPLE_SRC_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname})
  target_link_libraries(${name} ${SAMPLE_LIBS} dl rt atomic m)
  #strip
  if(NOT BUILD_SHARED)
    add_custom_command(TARGET ${name} POST_BUILD
          COMMAND ${CROSS_COMPILE}strip $<TARGET_FILE:${name}>
          COMMENT "Stripping binary for ${name}")
  endif()
  install(TARGETS ${name} DESTINATION bin/c)
endforeach ()
