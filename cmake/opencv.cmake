project(opencv_fetchcontent)

# Get the architecture-specific part based on the toolchain file
if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-cvitek-linux-uclibcgnueabihf.cmake")
  set(ARCHITECTURE "uclibc")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-linux-gnueabihf.cmake")
  set(ARCHITECTURE "32bit")
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
else()
  message(FATAL_ERROR "No shrinked opencv library for ${CMAKE_TOOLCHAIN_FILE}")
endif()


if (IS_LOCAL)
  set(OPENCV_URL ${3RD_PARTY_URL_PREFIX}${ARCHITECTURE}/opencv_aisdk.tar.gz)
else()
  set(OPENCV_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/opencv_aisdk.tar.gz)
endif()


if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/opencv-src/lib")
  FetchContent_Declare(
    opencv
    URL ${OPENCV_URL}
  )
  FetchContent_MakeAvailable(opencv)
  message("Content downloaded to ${opencv_SOURCE_DIR}")
endif()
set(OPENCV_ROOT ${BUILD_DOWNLOAD_DIR}/opencv-src)

set(OPENCV_INCLUDES
  ${OPENCV_ROOT}/include/
  ${OPENCV_ROOT}/include/opencv/
)

set(OPENCV_LIBS_IMCODEC ${OPENCV_ROOT}/lib/libopencv_core.so
                        ${OPENCV_ROOT}/lib/libopencv_imgproc.so
                        ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so)

set(OPENCV_LIBS_IMCODEC_STATIC ${OPENCV_ROOT}/lib/libopencv_core.a
                               ${OPENCV_ROOT}/lib/libopencv_imgproc.a
                               ${OPENCV_ROOT}/lib/libopencv_imgcodecs.a)
if (NOT "${ARCH}" STREQUAL "riscv")
  set(OPENCV_LIBS_IMCODEC_STATIC ${OPENCV_LIBS_IMCODEC_STATIC}
                          ${OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libtegra_hal.a)
endif()

set(OPENCV_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/opencv)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_core.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_core.so)
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_imgproc.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_imgproc.so)
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_imgcodecs.so)
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_core.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_core.so.3.2)
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_imgproc.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_imgproc.so.3.2)
install(PROGRAMS ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so.3.2.0 DESTINATION ${OPENCV_PATH}/lib RENAME libopencv_imgcodecs.so.3.2)
else()
file(GLOB OPENCV_LIBS "${OPENCV_ROOT}/lib/*so*")
install(FILES ${OPENCV_LIBS} DESTINATION ${OPENCV_PATH}/lib)
endif()
install(FILES ${OPENCV_LIBS_IMCODEC_STATIC} DESTINATION ${OPENCV_PATH}/lib)
install(DIRECTORY ${OPENCV_ROOT}/include/ DESTINATION ${OPENCV_PATH}/include)
