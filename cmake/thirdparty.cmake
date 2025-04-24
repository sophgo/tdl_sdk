project(thirdparty_fetchcontent)

# Get the architecture-specific part based on the toolchain file
if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-cvitek-linux-uclibcgnueabihf.cmake")
  set(ARCHITECTURE "uclibc")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-linux-gnueabihf.cmake")
  set(ARCHITECTURE "32bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-gnueabihf.cmake")
  set(ARCHITECTURE "32bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-musleabihf.cmake")
  set(ARCHITECTURE "musl")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-linux-gnu.cmake")
  set(ARCHITECTURE "64bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-none-linux-gnu.cmake")
  set(ARCHITECTURE "64bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-buildroot-linux-gnu.cmake")
  set(ARCHITECTURE "64bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "riscv64-unknown-linux-gnu.cmake")
  set(ARCHITECTURE "glibc_riscv64")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "riscv64-unknown-linux-musl.cmake")
  set(ARCHITECTURE "musl_riscv64")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "x86_64-linux-gnu.cmake")
  set(ARCHITECTURE "x86_64")
else()
  message(FATAL_ERROR "No shrinked 3rd party library for ${CMAKE_TOOLCHAIN_FILE}")
endif()

if (IS_LOCAL)
  set(EIGEN_URL ${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/eigen.tar.gz)
else()
  set(EIGEN_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/eigen.tar.gz)
endif()
message("EIGEN_URL:${EIGEN_URL},is_local:${IS_LOCAL}")

if (NOT IS_DIRECTORY  "${BUILD_DOWNLOAD_DIR}/libeigen-src")
  FetchContent_Declare(
    libeigen
    URL ${EIGEN_URL}
  )
  FetchContent_MakeAvailable(libeigen)
  message("Content downloaded to ${libeigen_SOURCE_DIR}")
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/libeigen-src/include/eigen3)

if (IS_LOCAL)
  set(GOOGLETEST_URL ${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/googletest.tar.gz)
else()
  set(GOOGLETEST_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/googletest.tar.gz)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "Build GMOCK")
set(INSTALL_GTEST OFF CACHE BOOL "Install GMOCK")
if (NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/googletest-src")
  FetchContent_Declare(
    googletest
    URL ${GOOGLETEST_URL}
  )
  FetchContent_MakeAvailable(googletest)
  message("Content downloaded to ${googletest_SOURCE_DIR}")
else()
  project(googletest)
  add_subdirectory(${BUILD_DOWNLOAD_DIR}/googletest-src/)
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/googletest-src/googletest/include/gtest)

if (IS_LOCAL)
  set(NLOHMANNJSON_URL ${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/nlohmannjson.tar.gz)
else()
  set(NLOHMANNJSON_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/nlohmannjson.tar.gz)
endif()

if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/nlohmannjson-src")
  FetchContent_Declare(
    nlohmannjson
    URL ${NLOHMANNJSON_URL}
  )
  FetchContent_MakeAvailable(nlohmannjson)
  message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/nlohmannjson-src)

if(${CVI_PLATFORM} STREQUAL "BM1688")
  return()
endif()

if (IS_LOCAL)
  set(STB_URL ${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/stb.tar.gz)
else()
  set(STB_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/stb.tar.gz)
endif()

if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/stb-src")
  FetchContent_Declare(
    stb
    URL ${STB_URL}
  )
  FetchContent_MakeAvailable(stb)
  message("Content downloaded to ${stb_SOURCE_DIR}")
endif()
set(stb_SOURCE_DIR ${BUILD_DOWNLOAD_DIR}/stb-src)
include_directories(${stb_SOURCE_DIR})

