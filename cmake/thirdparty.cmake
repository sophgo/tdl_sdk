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

if("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
  #--------------libwebsockets--------------
  set(LIBWEBSOCKETS_TGZ "${BUILD_DOWNLOAD_DIR}/libwebsockets.tar.gz")
  set(LIBWEBSOCKETS_DST "${BUILD_DOWNLOAD_DIR}/libwebsockets-src")
  if(IS_LOCAL)
    # 在线环境：FTP 下载
    file(DOWNLOAD "${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/libwebsockets.tar.gz" "${LIBWEBSOCKETS_TGZ}"
        STATUS _dl_stat)
    list(GET _dl_stat 0 _rc)
    if(_rc)
      message(FATAL_ERROR "Download libwebsockets.tar.gz failed: ${_dl_stat}")
    endif()
  else()
    # 离线/本地环境：直接复制
    file(COPY "${TOP_DIR}/tdl_sdk/dependency/thirdparty/libwebsockets.tar.gz"
        DESTINATION "${BUILD_DOWNLOAD_DIR}")
  endif()

  # ---------- 解包（只做一次） ----------
  if(NOT IS_DIRECTORY "${LIBWEBSOCKETS_DST}")
    file(ARCHIVE_EXTRACT
        INPUT       "${LIBWEBSOCKETS_TGZ}"
        DESTINATION "${LIBWEBSOCKETS_DST}")
  endif()

  set(LIBWEBSOCKETS_INCLUDE ${LIBWEBSOCKETS_DST}/include)
  include_directories(${LIBWEBSOCKETS_DST}/include)
  set(LIBWEBSOCKETS_LIBRARY ${LIBWEBSOCKETS_DST}/lib/libwebsockets.so)
  set(LIBWEBSOCKETS_LIBRARY_STATIC ${LIBWEBSOCKETS_DST}/lib/libwebsockets.a)
  set(LIBWEBSOCKETS_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/libwebsockets)
  # 安装LIBWEBSOCKETS动态库
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
      # 只安装主要的动态库文件，并创建软链接
      install(PROGRAMS ${LIBWEBSOCKETS_DST}/lib/libwebsockets.so.17 DESTINATION ${LIBWEBSOCKETS_PATH}/lib RENAME libwebsockets.so)
      install(PROGRAMS ${LIBWEBSOCKETS_DST}/lib/libwebsockets.so.17 DESTINATION ${LIBWEBSOCKETS_PATH}/lib RENAME libwebsockets.so.17)
  else()
      # 安装所有动态库文件
      file(GLOB LIBWEBSOCKETS_DYNAMIC_LIBS "${LIBWEBSOCKETS_DST}/lib/*so*")
      install(FILES ${LIBWEBSOCKETS_DYNAMIC_LIBS} DESTINATION ${LIBWEBSOCKETS_PATH}/lib)
  endif()
  # 安装LIBWEBSOCKETS静态库
  install(FILES ${LIBWEBSOCKETS_LIBRARY_STATIC} DESTINATION ${LIBWEBSOCKETS_PATH}/lib)
  # 安装LIBWEBSOCKETS头文件
  install(DIRECTORY ${LIBWEBSOCKETS_DST}/include/ DESTINATION ${LIBWEBSOCKETS_PATH}/include)
  # install(DIRECTORY ${LIBWEBSOCKETS_DST}/include/libwebsockets/ DESTINATION ${LIBWEBSOCKETS_PATH}/include)


  #--------------openssl--------------
  set(OPENSSL_TGZ "${BUILD_DOWNLOAD_DIR}/openssl.tar.gz")
  set(OPENSSL_DST "${BUILD_DOWNLOAD_DIR}/openssl-src")
  if(IS_LOCAL)
    # 在线环境：FTP 下载
    file(DOWNLOAD "${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/openssl.tar.gz" "${OPENSSL_TGZ}"
        STATUS _dl_stat)
    list(GET _dl_stat 0 _rc)
    if(_rc)
      message(FATAL_ERROR "Download openssl.tar.gz failed: ${_dl_stat}")
    endif()
  else()
    # 离线/本地环境：直接复制
    file(COPY "${TOP_DIR}/tdl_sdk/dependency/thirdparty/openssl.tar.gz"
        DESTINATION "${BUILD_DOWNLOAD_DIR}")
  endif()

  # ---------- 解包（只做一次） ----------
  if(NOT IS_DIRECTORY "${OPENSSL_DST}")
    file(ARCHIVE_EXTRACT
        INPUT       "${OPENSSL_TGZ}"
        DESTINATION "${OPENSSL_DST}")
  endif()

  set(OPENSSL_INCLUDE ${OPENSSL_DST}/include)
  include_directories(${OPENSSL_DST}/include)
  set(OPENSSL_LIBRARY ${OPENSSL_DST}/lib/libssl.so ${OPENSSL_DST}/lib/libcrypto.so)

  #--------------curl--------------
  set(CURL_TGZ "${BUILD_DOWNLOAD_DIR}/curl.tar.gz")
  set(CURL_DST "${BUILD_DOWNLOAD_DIR}/curl-src")

  # ---------- 获取 curl.tar.gz ----------
  if(IS_LOCAL)
    # 在线环境：FTP 下载
    file(DOWNLOAD "${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/curl.tar.gz" "${CURL_TGZ}"
        STATUS _dl_stat)
    list(GET _dl_stat 0 _rc)
    if(_rc)
      message(FATAL_ERROR "Download curl.tar.gz failed: ${_dl_stat}")
    endif()
  else()
    # 离线/本地环境：直接复制
    file(COPY "${TOP_DIR}/tdl_sdk/dependency/thirdparty/curl.tar.gz"
        DESTINATION "${BUILD_DOWNLOAD_DIR}")
  endif()

  # ---------- 解包（只做一次） ----------
  if(NOT IS_DIRECTORY "${CURL_DST}")
    file(ARCHIVE_EXTRACT
        INPUT       "${CURL_TGZ}"
        DESTINATION "${CURL_DST}")
  endif()

  message(STATUS "curl unpacked to: ${CURL_DST}")


  set(CURL_INCLUDE ${CURL_DST}/include)
  include_directories(${CURL_DST}/include)
  set(CURL_LIBRARY ${CURL_DST}/lib/libcurl.so)
  set(CURL_LIBRARY_STATIC ${CURL_DST}/lib/libcurl.a)
  set(CURL_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/curl)
  # 安装CURL动态库
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
      # 只安装主要的动态库文件，并创建软链接
      install(PROGRAMS ${CURL_DST}/lib/libcurl.so.4.8.0 DESTINATION ${CURL_PATH}/lib RENAME libcurl.so)
      install(PROGRAMS ${CURL_DST}/lib/libcurl.so.4.8.0 DESTINATION ${CURL_PATH}/lib RENAME libcurl.so.4)
      install(PROGRAMS ${CURL_DST}/lib/libcurl.so.4.8.0 DESTINATION ${CURL_PATH}/lib RENAME libcurl.so.4.8.0)
  else()
      # 安装所有动态库文件
      file(GLOB CURL_DYNAMIC_LIBS "${CURL_DST}/lib/*so*")
      install(FILES ${CURL_DYNAMIC_LIBS} DESTINATION ${CURL_PATH}/lib)
  endif()
  # 安装CURL静态库
  install(FILES ${CURL_LIBRARY_STATIC} DESTINATION ${CURL_PATH}/lib)
  # 安装CURL头文件
  install(DIRECTORY ${CURL_DST}/include/ DESTINATION ${CURL_PATH}/include)

  set(COMMON_ZLIB_URL_PREFIX "ftp://${FTP_SERVER_NAME}:${FTP_SERVER_PWD}@${FTP_SERVER_IP}/sw_rls/third_party/latest/")
  if (IS_LOCAL)
    set(ZLIB_URL ${COMMON_ZLIB_URL_PREFIX}${ARCHITECTURE}/zlib.tar.gz)
  else()
    set(ZLIB_URL ${TOP_DIR}/tdl_sdk/dependency/thirdparty/zlib.tar.gz)
  endif()

  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/zlib-src/lib")
    FetchContent_Declare(
      zlib
      URL ${ZLIB_URL}
    )
    FetchContent_MakeAvailable(zlib)
    message("Zlib downloaded from ${ZLIB_URL} to ${zlib_SOURCE_DIR}")
  endif()
  set(ZLIB_ROOT ${BUILD_DOWNLOAD_DIR}/zlib-src)
  include_directories(${BUILD_DOWNLOAD_DIR}/zlib-src/include)
  set(ZLIB_LIBRARY ${ZLIB_ROOT}/lib/libz.so)
endif()

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

