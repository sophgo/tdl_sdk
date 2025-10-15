project(thirdparty_fetchcontent)

# Get the architecture-specific part based on the toolchain file
if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-cvitek-linux-uclibcgnueabihf.cmake")
  set(ARCHITECTURE "uclibc")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-linux-gnueabihf.cmake")
  set(ARCHITECTURE "32bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-gnueabihf.cmake")
  set(ARCHITECTURE "glibc_arm")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-musleabihf.cmake")
  set(ARCHITECTURE "musl_arm")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-none-linux-musl.cmake")
  set(ARCHITECTURE "musl_arm64")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-linux-gnu.cmake")
  set(ARCHITECTURE "64bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "aarch64-none-linux-gnu.cmake")
  set(ARCHITECTURE "glibc_arm64")
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

# ===============Eigen===============
if(EXISTS "${OSS_TARBALL_PATH}/eigen.tar.gz")
  set(EIGEN_URL ${OSS_TARBALL_PATH}/eigen.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/eigen.tar.gz")
  set(EIGEN_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/eigen.tar.gz)
elseif(IS_LOCAL)
  set(EIGEN_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/eigen.tar.gz)
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
# ===============Eigen===============

# ===============kissfft===============
if(EXISTS "${OSS_TARBALL_PATH}/kissfft.tar.gz")
  set(KISSFFT_URL ${OSS_TARBALL_PATH}/kissfft.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/kissfft.tar.gz")
  set(KISSFFT_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/kissfft.tar.gz)
elseif(IS_LOCAL)
  set(KISSFFT_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/kissfft.tar.gz)
else()
  set(KISSFFT_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/kissfft.tar.gz)
endif()

if (NOT IS_DIRECTORY  "${BUILD_DOWNLOAD_DIR}/kissfft-src")
  FetchContent_Declare(
    kissfft
    URL ${KISSFFT_URL}
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # 确保全局PIC设置
  set(KISSFFT_TEST OFF CACHE BOOL "KISSFFT TEST")
  set(KISSFFT_TOOLS OFF CACHE BOOL "KISSFFT TOOLS")
  set(KISSFFT_PKGCONFIG OFF CACHE BOOL "KISSFFT PKGCONFIG")
  set(KISSFFT_STATIC ON CACHE BOOL "KISSFFT STATIC")

  FetchContent_MakeAvailable(kissfft)
  
  # 强制设置目标属性
  if(TARGET kissfft)
    set_target_properties(kissfft PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(kissfft PRIVATE -fPIC)
    endif()
  endif()
  
else()
  project(kissfft)
  add_subdirectory(${BUILD_DOWNLOAD_DIR}/kissfft-src
                   ${BUILD_DOWNLOAD_DIR}/kissfft-build)
  
  # 对已存在的目标设置属性
  if(TARGET kissfft)
    set_target_properties(kissfft PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(kissfft PRIVATE -fPIC)
    endif()
  endif()
endif()

set(KISSFFT_INCLUDES ${BUILD_DOWNLOAD_DIR}/kissfft-src)
# ===============kissfft===============


# ===============kaldi native fbank===============
if(EXISTS "${OSS_TARBALL_PATH}/kaldi-native-fbank.tar.gz")
  set(KALDI_URL ${OSS_TARBALL_PATH}/kaldi-native-fbank.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/kaldi-native-fbank.tar.gz")
  set(KALDI_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/kaldi-native-fbank.tar.gz)
elseif(IS_LOCAL)
  set(KALDI_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/kaldi-native-fbank.tar.gz)
else()
  set(KALDI_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/kaldi-native-fbank.tar.gz)
endif()

if (NOT IS_DIRECTORY  "${BUILD_DOWNLOAD_DIR}/kaldi-native-fbank-src")
  FetchContent_Declare(
    kaldi-native-fbank
    URL ${KALDI_URL}
  )
  FetchContent_MakeAvailable(kaldi-native-fbank)
  message("Content downloaded to ${kaldi-native-fbank_SOURCE_DIR}")
else()
  project(kaldi-native-fbank-src)
  add_subdirectory(${BUILD_DOWNLOAD_DIR}/kaldi-native-fbank-src/)

endif()

set(KISSFFT_STATIC_LIB  "${BUILD_DOWNLOAD_DIR}/kissfft-build/libkissfft-float.a" CACHE FILEPATH "" FORCE)
target_include_directories(kaldi-native-fbank-core PUBLIC ${KISSFFT_INCLUDES})
target_link_libraries(kaldi-native-fbank-core ${KISSFFT_STATIC_LIB})

set(FBANK_INCLUDES ${BUILD_DOWNLOAD_DIR}/kaldi-native-fbank-src)
# ===============kaldi native fbank===============


# ===============Google Test===============
if(EXISTS "${OSS_TARBALL_PATH}/googletest.tar.gz")
  set(GOOGLETEST_URL ${OSS_TARBALL_PATH}/googletest.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/googletest.tar.gz")
  set(GOOGLETEST_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/googletest.tar.gz)
elseif(IS_LOCAL)
  set(GOOGLETEST_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/googletest.tar.gz)
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

set(GTEST_INCLUDES ${BUILD_DOWNLOAD_DIR}/googletest-src/googletest/include)

# include_directories(${BUILD_DOWNLOAD_DIR}/googletest-src/googletest/include/gtest)
# ===============Google Test===============

# ===============nlohmannjson===============
if(EXISTS "${OSS_TARBALL_PATH}/nlohmannjson.tar.gz")
  set(NLOHMANNJSON_URL ${OSS_TARBALL_PATH}/nlohmannjson.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/nlohmannjson.tar.gz")
  set(NLOHMANNJSON_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/nlohmannjson.tar.gz)
elseif(IS_LOCAL)
  set(NLOHMANNJSON_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/nlohmannjson.tar.gz)
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
# ===============nlohmannjson===============

if(NOT "${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X" AND NOT "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  # ===============libwebsockets===============
  if(EXISTS "${OSS_TARBALL_PATH}/libwebsockets.tar.gz")
    set(LIBWEBSOCKETS_URL ${OSS_TARBALL_PATH}/libwebsockets.tar.gz)
  elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/libwebsockets.tar.gz")
    set(LIBWEBSOCKETS_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/libwebsockets.tar.gz)
  elseif(IS_LOCAL)
    set(LIBWEBSOCKETS_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/libwebsockets.tar.gz)
  else()
    set(LIBWEBSOCKETS_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/libwebsockets.tar.gz)
  endif()
  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/libwebsockets-src")
    FetchContent_Declare(
      libwebsockets
      URL ${LIBWEBSOCKETS_URL}
    )
    FetchContent_MakeAvailable(libwebsockets)
    message("Content downloaded from ${LIBWEBSOCKETS_URL} to ${libwebsockets_SOURCE_DIR}")
  endif()
  set(LIBWEBSOCKETS_ROOT ${BUILD_DOWNLOAD_DIR}/libwebsockets-src)
  include_directories(${LIBWEBSOCKETS_ROOT}/include)
  set(LIBWEBSOCKETS_LIBS ${LIBWEBSOCKETS_ROOT}/lib/libwebsockets.so)
  set(LIBWEBSOCKETS_LIBS_STATIC ${LIBWEBSOCKETS_ROOT}/lib/libwebsockets.a)
  set(LIBWEBSOCKETS_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/libwebsockets)
  file(GLOB LIBWEBSOCKETS_LIBS_INSTALL "${LIBWEBSOCKETS_ROOT}/lib/libwebsockets.so*")
  install(FILES ${LIBWEBSOCKETS_LIBS_INSTALL} DESTINATION ${LIBWEBSOCKETS_PATH}/lib)

  install(DIRECTORY ${LIBWEBSOCKETS_ROOT}/include/ DESTINATION ${LIBWEBSOCKETS_PATH}/include)
  # ===============libwebsockets===============

  # ===============openssl===============
  if(EXISTS "${OSS_TARBALL_PATH}/openssl.tar.gz")
    set(OPENSSL_URL ${OSS_TARBALL_PATH}/openssl.tar.gz)
  elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/openssl.tar.gz")
    set(OPENSSL_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/openssl.tar.gz)
  elseif(IS_LOCAL)
    set(OPENSSL_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/openssl.tar.gz)
  else()
    set(OPENSSL_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/openssl.tar.gz)
  endif()
  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/openssl-src")
    FetchContent_Declare(
      openssl
      URL ${OPENSSL_URL}
    )
    FetchContent_MakeAvailable(openssl)
    message("Content downloaded from ${OPENSSL_URL} to ${openssl_SOURCE_DIR}")
  endif()
  set(OPENSSL_ROOT ${BUILD_DOWNLOAD_DIR}/openssl-src)
  include_directories(${OPENSSL_ROOT}/include)
  set(OPENSSL_LIBRARY ${OPENSSL_ROOT}/lib/libssl.so ${OPENSSL_ROOT}/lib/libcrypto.so)
  set(OPENSSL_LIBRARY_STATIC ${OPENSSL_ROOT}/lib/libssl.a ${OPENSSL_ROOT}/lib/libcrypto.a)
  set(OPENSSL_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/openssl)
  file(GLOB OPENSSL_LIBS_INSTALL "${OPENSSL_ROOT}/lib/libssl.so*" "${OPENSSL_ROOT}/lib/libcrypto.so*")
  install(FILES ${OPENSSL_LIBS_INSTALL} DESTINATION ${OPENSSL_PATH}/lib)
  install(DIRECTORY ${OPENSSL_ROOT}/include/ DESTINATION ${OPENSSL_PATH}/include)
  # ===============openssl===============

  # ===============curl===============
  if(EXISTS "${OSS_TARBALL_PATH}/curl.tar.gz")
    set(CURL_URL ${OSS_TARBALL_PATH}/curl.tar.gz)
  elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/curl.tar.gz")
    set(CURL_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/curl.tar.gz)
  elseif(IS_LOCAL)
    set(CURL_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/curl.tar.gz)
  else()
    set(CURL_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/curl.tar.gz)
  endif()
  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/curl-src")
    FetchContent_Declare(
      curl
      URL ${CURL_URL}
    )
    FetchContent_MakeAvailable(curl)
    message("Content downloaded from ${CURL_URL} to ${curl_SOURCE_DIR}")
  endif()
  set(CURL_ROOT ${BUILD_DOWNLOAD_DIR}/curl-src)
  include_directories(${CURL_ROOT}/include)
  set(CURL_LIBRARY ${CURL_ROOT}/lib/libcurl.so)
  set(CURL_LIBRARY_STATIC ${CURL_ROOT}/lib/libcurl.a)
  set(CURL_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/curl)
  file(GLOB CURL_LIBS_INSTALL "${CURL_ROOT}/lib/libcurl.so*")
  install(FILES ${CURL_LIBS_INSTALL} DESTINATION ${CURL_PATH}/lib)
  install(DIRECTORY ${CURL_ROOT}/include/ DESTINATION ${CURL_PATH}/include)
  # ===============curl===============

  # ===============zlib===============
  if(EXISTS "${OSS_TARBALL_PATH}/zlib.tar.gz")
    set(ZLIB_URL ${OSS_TARBALL_PATH}/zlib.tar.gz)
  elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/zlib.tar.gz")
    set(ZLIB_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/zlib.tar.gz)
  elseif(IS_LOCAL)
    set(ZLIB_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/zlib.tar.gz)
  else()
    set(ZLIB_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/zlib.tar.gz)
  endif()
  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/zlib-src")
    FetchContent_Declare(
      zlib
      URL ${ZLIB_URL}
    )
    FetchContent_MakeAvailable(zlib)
    message("Zlib downloaded from ${ZLIB_URL} to ${zlib_SOURCE_DIR}")
  endif()
  set(ZLIB_ROOT ${BUILD_DOWNLOAD_DIR}/zlib-src)
  include_directories(${ZLIB_ROOT}/include)
  set(ZLIB_LIBRARY ${ZLIB_ROOT}/lib/libz.so)
  set(ZLIB_LIBRARY_STATIC ${ZLIB_ROOT}/lib/libz.a)
  set(ZLIB_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/zlib)
  file(GLOB ZLIB_LIBS_INSTALL "${ZLIB_ROOT}/lib/libz.so*")
  install(FILES ${ZLIB_LIBS_INSTALL} DESTINATION ${ZLIB_PATH}/lib)
  install(DIRECTORY ${ZLIB_ROOT}/include/ DESTINATION ${ZLIB_PATH}/include)
  # ===============zlib===============
endif()

if(${CVI_PLATFORM} STREQUAL "BM1688")
  return()
endif()

if(EXISTS "${OSS_TARBALL_PATH}/stb.tar.gz")
  set(STB_URL ${OSS_TARBALL_PATH}/stb.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/stb.tar.gz")
  set(STB_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/stb.tar.gz)
elseif(IS_LOCAL)
  set(STB_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/stb.tar.gz)
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

