# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

cmake_minimum_required(VERSION 3.2.2)

project(qnn)
include(VERSION.cmake)

# CMake custom functions
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set utils
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/utils.cmake)

# Set CMake options
include(cmake/options.cmake)

# Update submodule
include(cmake/submodule.cmake)

#############################################################################################
# Not defined check
if(NOT DEFINED CMAKE_BOARD_TYPE)
    message(FATAL_ERROR "Board type is not set.")
elseif(CMAKE_BOARD_TYPE STREQUAL "bm1880")
    add_definitions(-DNPU_INT8)
else()
    add_definitions(-DNPU_FLOAT32)
    message(FATAL_ERROR "Board type not supported yet.")
endif()

# Must define TOOLCHAIN_ROOT_DIR if CMAKE_TOOLCHAIN_FILE is set
if(CMAKE_TOOLCHAIN_FILE)
    if(NOT DEFINED TOOLCHAIN_ROOT_DIR)
        message(FATAL_ERROR "TOOLCHAIN_ROOT_DIR is not set. Aborting.")
    endif()
    set(USE_TOOLCHAIN "true")
else()
    set(USE_TOOLCHAIN "false")
endif()

# Get OS version
execute_process(COMMAND lsb_release -sr OUTPUT_VARIABLE HOST_OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${PLATFORM}" STREQUAL "")
    message(FATAL_ERROR "PLATFORM is not set (cmodel, usb, soc). Aborting.")
elseif("${PLATFORM}" STREQUAL "soc")
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(3rdparty/nlohmann_json)
    set(BM_NETWORKS_TARGET_BASENAME "soc_${CMAKE_BOARD_TYPE}_asic" CACHE INTERNAL "" FORCE)
elseif("${PLATFORM}" STREQUAL "cmodel")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
    find_package(Glog REQUIRED)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(3rdparty/nlohmann_json)
    set(BM_NETWORKS_TARGET_BASENAME "cmodel_${CMAKE_BOARD_TYPE}" CACHE INTERNAL "" FORCE)
elseif("${PLATFORM}" STREQUAL "usb")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
    find_package(Glog REQUIRED)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(3rdparty/nlohmann_json)
    set(BM_NETWORKS_TARGET_BASENAME "usb_${CMAKE_BOARD_TYPE}" CACHE INTERNAL "" FORCE)
endif()

# Currently the BSP release has the different name of bmmet_get_input_threshold instead of bmnet_get_input_threshold
# temperally use complie flag to WA it
if(USE_TOOLCHAIN)
    if(DEFINED BSPSDK_ROOT_DIR AND NOT BSPSDK_ROOT_DIR STREQUAL "")
        set(USE_BSPSDK "true")
        #add_definitions(-DUSE_BSPSDK)
    else()
        set(USE_BSPSDK "false")
    endif()
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(WARNING "CMAKE_BUILD_TYPE is not set. Use default configuration Release.")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Set prebuilt library path
if(NOT DEFINED PREBUILT_DIR OR "${PREBUILT_DIR}" STREQUAL "")
    set(PREBUILT_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME})
endif()

# Set install path
set(BM_INSTALL_INCLUDENAME "bm_networks")
if(CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "Default install path" FORCE)
elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "Default install path" FORCE)
endif()

# Set compile flags
set(CMAKE_C_INIT "-fsigned-char -Werror=all -fdiagnostics-color=always")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_INIT} -O3" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_INIT} -O3" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set( CMAKE_C_FLAGS "${CMAKE_C_INIT} -g -O0" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_INIT} -g -O0" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Asan")
    set( CMAKE_C_FLAGS "${CMAKE_C_INIT} -g -O0 -fsanitize=address" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_INIT} -g -O0 -fsanitize=address" )
else()
    message(FATAL_ERROR "No build type!!!")
endif()

# Set build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(BM_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${BM_NETWORKS_TARGET_BASENAME}/${CMAKE_BUILD_TYPE})

# Set config
configure_file(${CMAKE_SOURCE_DIR}/config.h.in config.h)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add 3rdparty librarires
include(3rdparty/tracer/tracer.cmake)

# Dependencies library cmake
include(cmake/libdep.cmake)

include(3rdparty/LibLinear/liblinear.cmake)

# Build Openpose Lite
add_subdirectory(3rdparty/openpose_lite)

#############################################################################################

message("======================================================")
message("[Build summary]")
message("    Board type: ${CMAKE_BOARD_TYPE}")
message("    Use toolchain: ${USE_TOOLCHAIN}")
message("    Use BSP Released SDK: ${USE_BSPSDK}")
message("    BSP SDK Path: ${BSPSDK_ROOT_DIR}")
message("    Build type: ${CMAKE_BUILD_TYPE}")
message("    CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("")
message("[Config]")
message("    CONFIG_FUNCTION_TRACE  ${CONFIG_FUNCTION_TRACE}")
message("    CONFIG_BUILD_SOC       ${CONFIG_BUILD_SOC}")
message("")
message("[Build]")
message("    BUILD_TEST             ${BUILD_TEST}")
message("    BUILD_DOC:             ${BUILD_DOC}")
message("")
message("[Prebuilt]")
message("    OpenCV version: ${OpenCV_VERSION}")
message("======================================================")


# Build networks
add_subdirectory("networks/utils/")
add_subdirectory("networks/core/")
add_subdirectory("networks/face/")

# Build test
if(BUILD_TEST)
    enable_testing()
    add_subdirectory("test")
endif()

# Build doc
if(BUILD_DOC)
    if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
        add_subdirectory("doc")
    else()
        message(WARNING "Doc is only available in Release mode.")
    endif()
endif()

# Export package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${BM_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${QNN_VERSION_STRING}
    COMPATIBILITY SameMajorVersion
)

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/qnnConfig.cmake.in
    "${BM_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake" @ONLY)
