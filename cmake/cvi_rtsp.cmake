include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(FetchContent)

if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-uclibc-linux.cmake")
  set(RTSP_SDK_VER "uclibc")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-aarch64-linux.cmake")
  set(RTSP_SDK_VER "64bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-gnueabihf-linux.cmake")
  set(RTSP_SDK_VER "32bit")
else()
  message(FATAL_ERROR "Unrecognized toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

ExternalProject_Add(cvi_rtsp
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/cvi_rtsp-Download
  GIT_REPOSITORY ssh://10.58.65.11:29418/cvi_rtsp
  BUILD_COMMAND CROSS_COMPILE=${TC_PATH}${CROSS_COMPILE} SDK_VER=${RTSP_SDK_VER} ./build.sh
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_IN_SOURCE true
  BUILD_BYPRODUCTS <SOURCE_DIR>/src/libcvi_rtsp.so
)

ExternalProject_Get_property(cvi_rtsp SOURCE_DIR)

set(cvi_rtsp_LIBPATH ${SOURCE_DIR}/src/libcvi_rtsp.so)
set(cvi_rtsp_INCLUDE ${SOURCE_DIR}/include/cvi_rtsp)

install(FILES ${cvi_rtsp_LIBPATH} DESTINATION lib)
install(DIRECTORY ${cvi_rtsp_INCLUDE} DESTINATION include)