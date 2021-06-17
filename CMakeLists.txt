cmake_minimum_required(VERSION 3.18.4)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(cviai)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install_soc")
endif()
# Find toolchain cmake file and toolchain folder
if("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
  message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Aborting.")
endif()

if("${CVI_PLATFORM}" STREQUAL "")
  set(CVI_PLATFORM "CV183X")
  message(AUTHOR_WARNING "Platform not provided, set to ${CVI_PLATFORM}.")
endif()

if(NOT ("${CVI_PLATFORM}" STREQUAL "CV183X" OR "${CVI_PLATFORM}" STREQUAL "CV182X"))
  message(FATAL_ERROR "Unrecognized platform ${CVI_PLATFORM}.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_INIT "-fsigned-char -Werror=all -fdiagnostics-color=always -std=c11")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17 -pthread")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT}" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_INIT} -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_INIT} -s" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_INIT}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_INIT}" )
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

# FIXME: Temporarily workaround
set(SYSTRACE_FALLBACK ON)
add_definitions(-DSYSTRACE_FALLBACK)

# Get current commit
execute_process(COMMAND git show --oneline -s WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE CURRENT_COMMIT)
execute_process(COMMAND git rev-parse --abbrev-ref HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE LATEST_BRANCH)
string(STRIP ${LATEST_BRANCH} LATEST_BRANCH)

if ("${LATEST_BRANCH}" STREQUAL "master")
  string(REPLACE " " ";" WORD_LIST ${CURRENT_COMMIT})
  list(GET WORD_LIST 0 COMMITID)
  set(LATEST_TAG "nightly-${COMMITID}")
else()
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)*[^(\\\n).]+" LATEST_TAG ${LATEST_BRANCH})
  if ("${LATEST_TAG}" STREQUAL "")
    set(LATEST_TAG "unknown")
  endif()
  string(REPLACE " " ";" WORD_LIST ${CURRENT_COMMIT})
  list(GET WORD_LIST 0 COMMITID)
  set(LATEST_TAG "${LATEST_TAG}-${COMMITID}")
endif()


message("==================================================")
message("Branch           ${LATEST_BRANCH}")
message("Latest tag       ${LATEST_TAG}")
message("Commit           ${CURRENT_COMMIT}")
message("[Summary]")
message("Build type       ${CMAKE_BUILD_TYPE}")
message("C   compiler     ${CMAKE_C_COMPILER}")
message("CXX compiler     ${CMAKE_CXX_COMPILER}")
message("C Flags          ${CMAKE_C_FLAGS}")
message("CXX Flags        ${CMAKE_CXX_FLAGS}")
message("Install dir      ${CMAKE_INSTALL_PREFIX}")
message("Enable Perfetto  ${ENABLE_PERFETTO}")
message("CVI_PLATFORM     ${CVI_PLATFORM}")
message("==================================================")

# Check if OpenCV exists.
include(cmake/opencv.cmake)

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if middleware exists.
include (cmake/middleware.cmake)

# Check if ive exists.
include (${IVE_SDK_ROOT}/cmake/ive.cmake)


include (cmake/cvi_rtsp.cmake)

# 3rdparty libraries
include(cmake/thirdparty.cmake)

# 3rdpart includes
include_directories(${IVE_INCLUDES}
                    ${OPENCV_INCLUDES}
                    ${MLIR_INCLUDES}
                    ${MIDDLEWARE_INCLUDES}
                    ${cvi_rtsp_INCLUDE})
add_subdirectory(modules)
add_subdirectory(sample)
add_subdirectory(regression)

# Install doc
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc DESTINATION doc)