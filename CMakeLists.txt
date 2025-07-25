cmake_minimum_required(VERSION 3.16.3)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
project(tdl_sdk)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wno-psabi)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# 使用并行编译
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
endif()

if(DEFINED ENV{USE_BMCV} AND "$ENV{USE_BMCV}" STREQUAL "1")
    add_definitions(-DUSE_BMCV)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Winvalid-pch)
    set_target_properties(${TARGET_NAME} PROPERTIES
        COMPILE_FLAGS "-include ${CMAKE_CURRENT_BINARY_DIR}/pch.hpp")
endif()

if("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")
endif()

# Find toolchain cmake file and toolchain folder
if( ("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "") AND (NOT "${CVI_PLATFORM}" STREQUAL "BM1688") )
  message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Aborting.")
else()
  message(STATUS "CMAKE_TOOLCHAIN_FILE:${CMAKE_TOOLCHAIN_FILE}")
endif()

set(CMAKE_C_INIT "-fsigned-char -fdiagnostics-color=always -std=gnu11")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -fdiagnostics-color=always -std=gnu++14 -pthread")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if("${CVI_PLATFORM}" STREQUAL "CV180X")
  add_definitions(-D__CV180X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV181X")
  add_definitions(-D__CV181X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV182X")
  add_definitions(-D__CV182X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV183X")
  add_definitions(-D__CV183X__)
  add_definitions(-DUSE_NEON)
elseif("${CVI_PLATFORM}" STREQUAL "CV184X")
  add_definitions(-D__CV184X__)
elseif("${CVI_PLATFORM}" STREQUAL "SOPHON")
  add_definitions(-D__CV186X__)
elseif("${CVI_PLATFORM}" STREQUAL "BM1688")
  add_definitions(-D__BM168X__)
  add_definitions(-D__BM1688__)
elseif("${CVI_PLATFORM}" STREQUAL "BM1684")
  add_definitions(-D__BM168X__)
  add_definitions(-D__BM1684__)  
elseif("${CVI_PLATFORM}" STREQUAL "BM1684X")
  add_definitions(-D__BM168X__)
  add_definitions(-D__BM1684X__)
elseif("${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X")
  add_definitions(-D__CMODEL_CV181X__)
elseif("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  add_definitions(-D__CMODEL_CV184X__)
else()
  message(FATAL_ERROR "Unrecognized platform ${CVI_PLATFORM}.")
endif()

if("${CVI_PLATFORM}" STREQUAL "BM1684" OR 
   "${CVI_PLATFORM}" STREQUAL "BM1684X" OR 
   "${CVI_PLATFORM}" STREQUAL "BM1688")
  add_definitions(-D__EDGE_PLATFORM__)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT} -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT} -s" )
    add_definitions(-DDISABLE_LOG)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT} -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT} -s" )
    add_definitions(-DDISABLE_LOG)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_INIT} -g" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_INIT} -g" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Asan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_INIT} -fsanitize=address" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_INIT} -fsanitize=address" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Tsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_INIT} -fsanitize=thread" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_INIT} -fsanitize=thread" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "UBsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_INIT} -fsanitize=undefined" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_INIT} -fsanitize=undefined" )
    set( CMAKE_EXE_LINKER_FLAGS "-lubsan")
else()
    message(FATAL_ERROR "No build type!!!")
endif()

get_directory_property(compile_defs COMPILE_DEFINITIONS)
message(STATUS "All compile definitions: ${compile_defs}")

set(IS_LOCAL FALSE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  # Get current commit
  execute_process(COMMAND git show --oneline -s WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE CURRENT_COMMIT)
  execute_process(COMMAND git rev-parse --abbrev-ref HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE LATEST_BRANCH)
  string(STRIP ${LATEST_BRANCH} LATEST_BRANCH)

  if("${LATEST_BRANCH}" STREQUAL "master")
    string(REPLACE " " ";" WORD_LIST ${CURRENT_COMMIT})
    list(GET WORD_LIST 0 COMMITID)
    set(LATEST_TAG "nightly-${COMMITID}")
  elseif("${LATEST_BRANCH}" STREQUAL "HEAD")
    string(REPLACE " " ";" WORD_LIST ${CURRENT_COMMIT})
    list(GET WORD_LIST 0 COMMITID)
    set(LATEST_TAG "${COMMITID}")
  else()
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)*[^(\\\n).]+" LATEST_TAG ${LATEST_BRANCH})
    if("${LATEST_TAG}" STREQUAL "")
      set(LATEST_TAG "unknown")
    endif()
    string(REPLACE " " ";" WORD_LIST ${CURRENT_COMMIT})
    list(GET WORD_LIST 0 COMMITID)
    set(LATEST_TAG "${LATEST_TAG}-${COMMITID}")
  endif()

  execute_process(
      COMMAND git remote -v
      OUTPUT_VARIABLE REMOTE_URL
      ERROR_VARIABLE GIT_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(FIND "${REMOTE_URL}" "cvitek" FOUND_POS)

  if(NOT ${FOUND_POS} EQUAL -1)
      set(IS_LOCAL TRUE)
  else()
      set(IS_LOCAL FALSE)
  endif()

endif()

if (NOT IS_LOCAL AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependency/thirdparty")

  execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_thirdparty.sh "${CMAKE_TOOLCHAIN_FILE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE DOWNLOAD_RESULT
  )
  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "download_thirdparty.sh failed: ${DOWNLOAD_RESULT}")
  endif()

endif()

foreach(flags CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
    string(REPLACE "-O3" "-Os" ${flags} "${${flags}}")
endforeach()

include(${CMAKE_TOOLCHAIN_FILE})

message("==================================================")
message("Branch                 ${LATEST_BRANCH}")
message("Latest tag             ${LATEST_TAG}")
message("Commit                 ${CURRENT_COMMIT}")
message("[Summary]")
message("Build type             ${CMAKE_BUILD_TYPE}")
message("C   compiler           ${CMAKE_C_COMPILER}")
message("CXX compiler           ${CMAKE_CXX_COMPILER}")
message("C Flags                ${CMAKE_C_FLAGS}")
message("CXX Flags              ${CMAKE_CXX_FLAGS}")
message("Install dir            ${CMAKE_INSTALL_PREFIX}")
message("CMAKE_TOOLCHAIN_FILE   ${CMAKE_TOOLCHAIN_FILE}")
message("Enable Perfetto        ${ENABLE_PERFETTO}")
message("CVI_PLATFORM           ${CVI_PLATFORM}")
message("USE_TPU_IVE            ${USE_TPU_IVE}")
message("FTP_SERVER_IP          ${FTP_SERVER_IP}")
message("CONFIG_DUAL_OS         ${CONFIG_DUAL_OS}")
message("==================================================")

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(FetchContent)
get_filename_component(_deps "${BUILD_DOWNLOAD_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${_deps})

# set 3rd_party url
set(3RD_PARTY_URL_PREFIX "ftp://${FTP_SERVER_NAME}:${FTP_SERVER_PWD}@${FTP_SERVER_IP}/sw_rls/third_party/latest/")

if("${CVI_PLATFORM}" STREQUAL "BM1684X" OR
   "${CVI_PLATFORM}" STREQUAL "BM1684" OR
   "${CVI_PLATFORM}" STREQUAL "BM1688" OR
   "${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X" OR
   "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  set(IS_LOCAL OFF)
endif()

# Default dynamic compilation
if(${BUILD_SHARED})
  set(BUILD_SHARED ON)
else()
  set(BUILD_SHARED OFF)
endif()

# open app modules use the video module for test
set(APP_VIDEO_ENABLE OFF)

set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

get_filename_component(TOP_DIR ${CURRENT_DIR} DIRECTORY)

set(REPO_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Check if OpenCV exists.
include(cmake/opencv.cmake)

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if mpi SDK exists.
if( NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X"))
  include(cmake/middleware.cmake)
endif()

# 3rdparty libraries
include(cmake/thirdparty.cmake)

include(cmake/ive.cmake)

# 3rdparty includes
include_directories(${IVE_INCLUDES}
                    ${OPENCV_INCLUDES}
                    ${MLIR_INCLUDES}
                    ${MIDDLEWARE_INCLUDES})

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if("${BUILD_OPTION}" STREQUAL "")
  add_subdirectory(src)
  add_subdirectory(sample)
elseif("${BUILD_OPTION}" STREQUAL "sample")
  add_subdirectory(sample)
elseif("${BUILD_OPTION}" STREQUAL "all")
  add_subdirectory(src)
  add_subdirectory(sample)
  if(IS_LOCAL)
    add_subdirectory(regression)
  endif()
  add_subdirectory(evaluation)
else()
  message(FATAL_ERROR "Error: BUILD_OPTION is set to an invalid value: ${BUILD_OPTION}.
                       Valid options are '', 'sample', or 'all', and 'clean'.")
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/configs DESTINATION .)