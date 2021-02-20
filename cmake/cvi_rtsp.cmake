include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(FetchContent)

if (ARCH STREQUAL "arm64")
  set(LIVE555_URL ftp://10.34.33.5/prebuilt/easy_build/live555/live555_2020.08.12_cv183x.tar.xz)
else()
  set(LIVE555_URL ftp://10.34.33.5/prebuilt/easy_build/live555/live555_2020.08.12_cv183x_lib32.tar.xz)
endif()

FetchContent_Declare(
  live555
  URL ${LIVE555_URL}
)
FetchContent_MakeAvailable(live555)
message("Content downloaded from ${LIVE555_URL} to ${live555_SOURCE_DIR}")

ExternalProject_Add(cvi_rtsp
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/cvi_rtsp-Download
  GIT_REPOSITORY ssh://10.58.65.11:29418/cvi_rtsp
  BUILD_COMMAND CROSS_COMPILE=${TC_PATH}${CROSS_COMPILE} LIVE555_DIR=${live555_SOURCE_DIR} ./build.sh
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_IN_SOURCE true
  BUILD_BYPRODUCTS <SOURCE_DIR>/src/libcvi_rtsp.so
)

ExternalProject_Get_property(cvi_rtsp SOURCE_DIR)

set(cvi_rtsp_LIBPATH ${SOURCE_DIR}/src/libcvi_rtsp.so)
set(cvi_rtsp_INCLUDE ${SOURCE_DIR}/include/cvi_rtsp)

install(FILES ${cvi_rtsp_LIBPATH} DESTINATION lib)
