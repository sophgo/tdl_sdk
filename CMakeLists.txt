cmake_minimum_required(VERSION 3.2.2)

# Options
# option(ENABLE_SYSTRACE "Use systrace" OFF)

if (CMAKE_C_COMPILER STREQUAL "/usr/bin/cc")
  message(FATAL_ERROR "Please use clang compiler. Please export C and CXX compiler to clang.")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if("${CVI_PLATFORM}" STREQUAL "")
  set(CVI_PLATFORM "cv1835")
  message(AUTHOR_WARNING "Platform not provided, set to ${CVI_PLATFORM}.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if("${CVI_TARGET}" STREQUAL "soc")
  add_definitions(-DCVI_SOC)
  if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install_soc")
  endif()
  # Find toolchain cmake file and toolchain folder
  if("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Aborting.")
  endif()
  if(NOT DEFINED TOOLCHAIN_ROOT_DIR)
    message(FATAL_ERROR "TOOLCHAIN_ROOT_DIR is not set. Aborting.")
  endif()
  set(USE_TOOLCHAIN "true")
  set(SSE_FLAGS "")
  set(GCC_STRIP_FLAG "-s")
else()
  set(CVI_TARGET "cmodel")
  if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")
  endif()
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(USE_TOOLCHAIN "false")
  set(SSE_FLAGS "-mssse3 -msse4.1 -mavx2 -mfma")
  set(GCC_STRIP_FLAG "")
endif()

add_definitions(-DENABLE_TRACE)

# Check if OpenCV exists.
include(cmake/opencv.cmake)

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if middleware exists.
include (cmake/middleware.cmake)

# Check if ive exists.
include (${IVE_SDK_ROOT}/cmake/ive.cmake)

# Check if Tracer SDK exists.
include(cmake/tracer.cmake)

set(CMAKE_C_INIT "-fsigned-char -Werror=all -fdiagnostics-color=always -std=c11 ${SSE_FLAGS}")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17 ${SSE_FLAGS}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -O3" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -O3" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -O3 ${GCC_STRIP_FLAG}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -O3 ${GCC_STRIP_FLAG}" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -g -O0" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -g -O0" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Asan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -g -O0 -fsanitize=address" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -g -O0 -fsanitize=address" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Tsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -g -O0 -fsanitize=thread" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -g -O0 -fsanitize=thread" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "UBsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -g -O0 -fsanitize=undefined" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -g -O0 -fsanitize=undefined" )
    set( CMAKE_EXE_LINKER_FLAGS "-lubsan")
else()
    message(FATAL_ERROR "No build type!!!")
endif()

# Get current commit
execute_process(COMMAND git show --oneline -s OUTPUT_VARIABLE CURRENT_COMMIT)

message("==================================================")
message("Commit           ${CURRENT_COMMIT}")
message("[Summary]")
message("Platform         ${CVI_PLATFORM}")
message("Target           ${CVI_TARGET}")
message("Build type       ${CMAKE_BUILD_TYPE}")
message("C   compiler     ${CMAKE_C_COMPILER}")
message("CXX compiler     ${CMAKE_CXX_COMPILER}")
message("C Flags          ${CMAKE_C_FLAGS}")
message("CXX Flags        ${CMAKE_CXX_FLAGS}")
message("Install dir      ${CMAKE_INSTALL_PREFIX}")
message("==================================================")


include_directories(${CMAKE_SOURCE_DIR}/3rdparty
                    ${CMAKE_SOURCE_DIR}/include
                    ${OPENCV_INCLUDES}
                    ${MLIR_INCLUDES}
                    ${MIDDLEWARE_INCLUDES}
                    ${TRACER_INCLUDES})
add_subdirectory(modules)
add_subdirectory(sample)
add_subdirectory(regression)
