cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(nnfactory)

option(BUILD_X86 "BUILD_ON_X86" OFF)
set(CXX11_FLAG "-std=c++11")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CXX_FLAG "-g -O0 -fPIC")
  message(STATUS "Build in Debug mode...")
else ()
  set(CXX_FLAG "-Wall -Wno-sign-compare -fopenmp -O3 -fPIC")
  message(STATUS "Build in Release mode...")
endif ()

set(CMAKE_CXX_FLAGS "${CXX_FLAG} ${CXX11_FLAG}")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

################################################################################################################
# platform build types
# all relevant settings and varibles should be set inside sub-cmake-file
#################################################################################################################
# There're total 3 building-platform types, more build type may be added in ther future,
# but each of them should have their own sub-cmake-files
#1: native_x86: build pcie on x86, for sc5
#2: native_arm: build soc on arm
#3: cross_arm:  arm soc cross  build on x86, for se3 or se5
#4: cross_arm_pcie: arm pcie cross build on x86, for phytium
if(PLATFORM_BUILD_TYPE STREQUAL native_x86)
    include("native_x86.cmake")
elseif(PLATFORM_BUILD_TYPE STREQUAL native_arm)
    include("native_arm.cmake")
elseif(PLATFORM_BUILD_TYPE STREQUAL cross_arm)
    include("cross_arm.cmake")
elseif(PLATFORM_BUILD_TYPE STREQUAL arm_pcie)
    include("arm_pcie.cmake")
else()
    message(FATAL_ERROR "Blank or unsupport platform build type! Check your PLATFORM_BUILD_BYPTE setting please")
endif()

#################################################################################################################
# common setting
#################################################################################################################
set(ALGONN_ROOT ${CMAKE_SOURCE_DIR}/../)
set(INCLUDE_DIRS_3RD
    ${CMAKE_SOURCE_DIR}/../include)
message(STATUS "opencv includes" ${OpenCV_INCLUDE_DIRS})
message(STATUS "opencv libraries" ${OpenCV_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${INCLUDE_DIRS_3RD} include ${OpenCV_INCLUDE_DIRS})

#################################################################################################################
# cmake targets
#################################################################################################################
# prepare for source code list and library list
file(GLOB PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.*)
add_library(nnfactory SHARED ${PROJECT_SOURCES})
target_link_libraries(nnfactory
    ${PYTHON_LIBRARIES}
    ${ALGONN_LIB}
    ${OpenCV_LIBRARIES})
set_target_properties(nnfactory PROPERTIES PREFIX "")
set_target_properties(nnfactory PROPERTIES SUFFIX ${EXTSUFFIX})
message(STATUS "nnlib " "${CMAKE_CURRENT_BINARY_DIR}/nnfactory${EXTSUFFIX}")

#################################################################################################################
# post build process
#################################################################################################################
#copy
add_custom_command(TARGET nnfactory POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_BINARY_DIR}/nnfactory${EXTSUFFIX}"
    "${CMAKE_CURRENT_SOURCE_DIR}/nnfactory${EXTSUFFIX}")

