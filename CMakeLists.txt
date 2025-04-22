cmake_minimum_required(VERSION 3.16.3)
project(tdl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

if("${CVI_PLATFORM}" STREQUAL "CV181X")
  add_definitions(-D__CV181X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV180X")
  add_definitions(-D__CV180X__)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set(NO_OPENCV ON)
    add_definitions(-DNO_OPENCV)
  endif()
elseif("${CVI_PLATFORM}" STREQUAL "CV182X")
  add_definitions(-D__CV182X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV183X")
  add_definitions(-D__CV183X__)
  add_definitions(-DUSE_NEON)
elseif("${CVI_PLATFORM}" STREQUAL "SOPHON")
  add_definitions(-D__CV186X__)
elseif("${CVI_PLATFORM}" STREQUAL "CV184X")
  add_definitions(-D__CV184X__)
elseif("${CVI_PLATFORM}" STREQUAL "BM1688")
  add_definitions(-D__BM168X__)
elseif("${CVI_PLATFORM}" STREQUAL "BM1684")
  add_definitions(-D__BM168X__)
  add_definitions(-D__BM1684__)  
elseif("${CVI_PLATFORM}" STREQUAL "BM1684X")
  add_definitions(-D__BM168X__)
  add_definitions(-D__BM1684X__)
elseif("${CVI_PLATFORM}" STREQUAL "CMODEL")
  add_definitions(-D__CMODEL__)
else()
  message(FATAL_ERROR "Unrecognized platform ${CVI_PLATFORM}.")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT} -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT} -s" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT} -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT} -s" )
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

if(DEFINED ENV{PERF_EVAL})
  add_definitions(-DPERF_EVAL)
  message("set PERF_EVAL")
endif()

get_directory_property(compile_defs COMPILE_DEFINITIONS)
message(STATUS "All compile definitions: ${compile_defs}")

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

if("${CVI_PLATFORM}" STREQUAL "BM1684X" OR "${CVI_PLATFORM}" STREQUAL "BM1684" OR "${CVI_PLATFORM}" STREQUAL "BM1688" OR "${CVI_PLATFORM}" STREQUAL "CMODEL_CVITEK")
  set(IS_LOCAL OFF)
endif()

# Default dynamic compilation
set(BUILD_SHARED ON)

set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

get_filename_component(TOP_DIR ${CURRENT_DIR} DIRECTORY)

message(STATUS "Current Directory: ${CURRENT_DIR}")
message(STATUS "Parent Directory: ${TOP_DIR}")

set(REPO_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# Check if OpenCV exists.
if (NOT DEFINED NO_OPENCV)
include(cmake/opencv.cmake)
endif()

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if mpi SDK exists.
if( NOT ("${CVI_PLATFORM}" STREQUAL "CMODEL"))
  include(cmake/middleware.cmake)
endif()

# 3rdparty libraries
include(cmake/thirdparty.cmake)

include(cmake/ive.cmake)

# 3rdpart includes
include_directories(${IVE_INCLUDES}
                    ${OPENCV_INCLUDES}
                    ${MLIR_INCLUDES}
                    ${MIDDLEWARE_INCLUDES})

message(STATUS "OPENCV_INCLUDES:{${OPENCV_INCLUDES}}")
message(STATUS "OPENCV_LIBS:{${OPENCV_LIBS}}")
message(STATUS "MLIR_INCLUDES:{${MLIR_INCLUDES}}")
message(STATUS "MLIR_LIBS:{${MLIR_LIBS}}")
message(STATUS "MIDDLEWARE_INCLUDES:{${MIDDLEWARE_INCLUDES}}")
message(STATUS "MIDDLEWARE_LIBS:{${MIDDLEWARE_LIBS}}")
message(STATUS "${CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE}")

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if("${BUILD_OPTION}" STREQUAL "")
  add_subdirectory(src)
  add_subdirectory(sample)
  add_subdirectory(regression)
elseif("${BUILD_OPTION}" STREQUAL "sample")
  add_subdirectory(sample)
elseif("${BUILD_OPTION}" STREQUAL "all" OR "${BUILD_OPTION}" STREQUAL "bm168x")
  add_subdirectory(src)
  add_subdirectory(sample)
  if(IS_LOCAL AND NOT DEFINED NO_OPENCV )
    add_subdirectory(regression)
  endif()
else()
  message(FATAL_ERROR "Error: BUILD_OPTION is set to an invalid value: ${BUILD_OPTION}. \
                      Valid options are '', 'sample', 'bm168x' or 'all', and 'clean'.")
endif()
