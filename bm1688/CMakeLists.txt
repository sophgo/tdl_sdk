cmake_minimum_required(VERSION 2.8)
project(FACE_SDK)

## Options
option(BUILD_X86 "Build on X86 Architecture." OFF)
option(BUILD_ARMV8 "Build on ARM Architecture." ON)

option(BUILD_LINUX "Build on Linux OS." ON)
option(BUILD_WINDOWS "Build on Windows OS." OFF)

option(BUILD_SHARED "Build Shared SDK COMMON library." ON)
option(BUILD_STATIC "Build Static SDK COMMON library." ON)


option(USE_OPENCV "Use OpenCV to do image process" ON)
option(TIME_PRINT "Print time" ON)


option(USE_FFMPEG "use ffmpeg" OFF)


if (USE_FFMPEG)
  add_definitions(-DUSE_FFMPEG)
endif ()

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
# common platform build varible could be set below
#################################################################################################################
## Compiler setting
message(STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  set(CXX11_FLAG "-std=c++11")
  if (BUILD_DEBUG)
    set(CXX_FLAG "-g -O -fPIC")
  else ()
    set(CXX_FLAG "-Wall -Wno-sign-compare -fopenmp -O3 -fPIC")
  endif ()
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(CXX11_FLAG "-std=c++11 -stdlib=libc++")
  set(CXX_FLAG "-Wall -O3 -fPIC")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CXX11_FLAG "")
  set(CXX_FLAG "/O2 /Ob2 /DNDEBUG /MT")
  set(CMAKE_CXX_FLAGS_RELEASE "/MT")
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd")

else ()
  message(FATAL_ERROR, "The compiler is unsupported.")
endif ()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAG} ${CXX11_FLAG}")
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${CXX_FLAG} ${CXX11_FLAG}")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_ASM_FLAGS: " ${CMAKE_ASM_FLAGS})


if (TIME_PRINT)
  add_definitions(-DTIME_PRINT)
endif ()


#show all compiling definitions
get_directory_property(DIRDEFS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
message(STATUS "Compile definitions: ${DIRDEFS}")



set(INC ${PROJECT_SOURCE_DIR}/include)
set(SRC ${PROJECT_SOURCE_DIR}/src)
set(EXAMPLE ${PROJECT_SOURCE_DIR}/examples)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${INC})

## Library source files setting
# NetCompact
set(NETCOMPACT_FILES
    ${SRC}/netcompact/tensor.cpp
    ${SRC}/netcompact/net.cpp
    ${SRC}/common/common.cpp
    ${SRC}/common/cv_utils.cpp
    ${SRC}/common/timer.cpp
    ${SRC}/common/matcher.cpp
    ${SRC}/common/device_info.cpp
    ${SRC}/common/obj_det_utils.cpp
    )



set(net_file bm1688net.cpp)
set(model_factory_file model_bm1688.cpp)

set(NETCOMPACT_FILES ${NETCOMPACT_FILES} ${SRC}/netcompact/net/${net_file} ${SRC}/factory/${model_factory_file})


# Framework src files
file(GLOB FRAMEWORK_FILES ${SRC}/framework/*.cpp)
# Face src files
file(GLOB FACE_SRC_FILES ${SRC}/face/*.cpp)
file(GLOB DET_SRC_FILES ${SRC}/detection/*.cpp)
file(GLOB REID_SRC_FILES ${SRC}/reid/*.cpp)

#decoder src files
file(GLOB LOG_SRC_FILES ${SRC}/log/*.cpp)

file(GLOB CLASSIFY_FILES ${SRC}/classify/*.cpp)

# SDK_COMMON files
set(LIB_SDK_COMMON_SRC_FILES
    ${NETCOMPACT_FILES}
    ${FACE_SRC_FILES}
    ${FRAMEWORK_FILES}
    ${DET_SRC_FILES}
    ${CLASSIFY_FILES}
    ${REID_SRC_FILES}
    ${LOG_SRC_FILES}
    )


set(SDK_COMMON_LIB_NAME "sdk_common_${OS_NAME}_${ARCH_NAME}")


message(STATUS "sdk common lib name: ${SDK_COMMON_LIB_NAME}")

set(SDK_COMMON_LINK_LIBS
    ${NNTOOLCHAIN_LIB}
    ${OpenCV_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    )


get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
  message(STATUS "included dir='${dir}'")
endforeach ()




add_library(${SDK_COMMON_LIB_NAME} SHARED ${LIB_SDK_COMMON_SRC_FILES})
set_target_properties(${SDK_COMMON_LIB_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(${SDK_COMMON_LIB_NAME} ${SDK_COMMON_LINK_LIBS})


get_target_property(link_sos ${SDK_COMMON_LIB_NAME} LINK_LIBRARIES)
foreach (dir ${link_sos})
  message(STATUS "${dir}")
endforeach ()

add_custom_command(TARGET ${SDK_COMMON_LIB_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_BINARY_DIR}/lib/lib${SDK_COMMON_LIB_NAME}.so"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib${SDK_COMMON_LIB_NAME}.so")

set(EXAMPLE_LINK_LIBS
    ${SDK_COMMON_LIB_NAME}
    ${OpenCV_LIBRARIES}
    ${BMNNSDK_ROOT}
    ${FFMPEG_LIBRARIES}
    pthread
    )


file(GLOB_RECURSE EXAMPLE_FILES ${EXAMPLE}/*.cpp)

message(STATUS ${EXAMPLE_FILES})

foreach (fname ${EXAMPLE_FILES})
  set(EXAMPLE_SRC_FILES ${EXAMPLE_SRC_FILES} ${fname})
  get_filename_component(name ${fname} NAME_WE)
  add_executable(${name} ${fname})
  target_link_libraries(${name} ${EXAMPLE_LINK_LIBS})

endforeach ()

