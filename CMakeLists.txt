cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
if(NOT DEFINED TOOLCHAIN_ROOT_DIR)
  message(FATAL_ERROR "TOOLCHAIN_ROOT_DIR is not set. Aborting.")
endif()

# Check if OpenCV exists.
include(cmake/opencv.cmake)

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if middleware exists.
include (cmake/middleware.cmake)

# Check if ive exists.
include (${IVE_SDK_ROOT}/cmake/ive.cmake)

# Perfetto
if (ENABLE_PERFETTO)
  add_definitions(-DENABLE_TRACE)
endif()
include(cmake/perfetto.cmake)


set(CMAKE_C_INIT "-fsigned-char -Werror=all -fdiagnostics-color=always -std=c11")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -O3" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -O3" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} -O3 -s" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} -O3 -s" )
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
message("Build type       ${CMAKE_BUILD_TYPE}")
message("C   compiler     ${CMAKE_C_COMPILER}")
message("CXX compiler     ${CMAKE_CXX_COMPILER}")
message("C Flags          ${CMAKE_C_FLAGS}")
message("CXX Flags        ${CMAKE_CXX_FLAGS}")
message("Install dir      ${CMAKE_INSTALL_PREFIX}")
message("Enable Perfetto  ${ENABLE_PERFETTO}")
message("==================================================")


include_directories(${CMAKE_SOURCE_DIR}/3rdparty
                    ${CMAKE_SOURCE_DIR}/include
                    ${IVE_INCLUDES}
                    ${OPENCV_INCLUDES}
                    ${MLIR_INCLUDES}
                    ${MIDDLEWARE_INCLUDES}
                    ${TRACER_INCLUDES})
add_subdirectory(modules)
add_subdirectory(sample)
add_subdirectory(regression)
