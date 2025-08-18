if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR ("${CVI_PLATFORM}" STREQUAL "BM1684"))

  if("${OPENCV_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "You must set OPENCV_ROOT_DIR before building IVE library.")
  endif()

  set(OPENCV_INCLUDES ${OPENCV_ROOT_DIR}/include/opencv4)
  set(OpenCV_LIB_DIR ${OPENCV_ROOT_DIR}/lib)
  set(OPENCV_LIBS_IMCODEC ${OpenCV_LIB_DIR}/libopencv_core.so ${OpenCV_LIB_DIR}/libopencv_imgproc.so
    ${OpenCV_LIB_DIR}/libopencv_highgui.so ${OpenCV_LIB_DIR}/libopencv_imgcodecs.so
    ${OpenCV_LIB_DIR}/libopencv_videoio.so ${OpenCV_LIB_DIR}/libopencv_calib3d.so
    ${OpenCV_LIB_DIR}/libopencv_flann.so ${OpenCV_LIB_DIR}/libopencv_features2d.so) 

  set(OPENCV_LIBS_IMCODEC_STATIC ${OpenCV_LIB_DIR}/libopencv_core.a ${OpenCV_LIB_DIR}/libopencv_imgproc.a
    ${OpenCV_LIB_DIR}/libopencv_highgui.a ${OpenCV_LIB_DIR}/libopencv_imgcodecs.a
    ${OpenCV_LIB_DIR}/libopencv_videoio.a ${OpenCV_LIB_DIR}/libopencv_calib3d.a
    ${OpenCV_LIB_DIR}/libopencv_flann.a ${OpenCV_LIB_DIR}/libopencv_features2d.a) 

  return()
endif()

if(("${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X") OR ("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X"))

  if("${OPENCV_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "You must set OPENCV_ROOT_DIR first.")
  endif()

  set(OPENCV_INCLUDES ${OPENCV_ROOT_DIR}/include)
  set(OpenCV_LIB_DIR ${OPENCV_ROOT_DIR}/lib)
  set(OPENCV_LIBS_IMCODEC ${OpenCV_LIB_DIR}/libopencv_core.so ${OpenCV_LIB_DIR}/libopencv_imgproc.so
    ${OpenCV_LIB_DIR}/libopencv_highgui.so ${OpenCV_LIB_DIR}/libopencv_imgcodecs.so)

  return()
endif()

# Get the architecture-specific part based on the toolchain file
if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-cvitek-linux-uclibcgnueabihf.cmake")
  set(ARCHITECTURE "uclibc")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-linux-gnueabihf.cmake")
  set(ARCHITECTURE "32bit")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-gnueabihf.cmake")
  set(ARCHITECTURE "glibc_arm")
elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "arm-none-linux-musleabihf.cmake")
  set(ARCHITECTURE "musl_arm")
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
  message(FATAL_ERROR "No shrinked opencv library for ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(EXISTS "${OSS_TARBALL_PATH}/opencv4.5.tar.gz")
  set(OPENCV_URL ${OSS_TARBALL_PATH}/opencv4.5.tar.gz)
elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/opencv4.5.tar.gz")
  set(OPENCV_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/opencv4.5.tar.gz)
elseif(IS_LOCAL)
  set(OPENCV_URL ${3RD_PARTY_URL_PREFIX}/${ARCHITECTURE}/opencv4.5.tar.gz)
else()
  message(FATAL_ERROR "Failed to find opencv4.5.tar.gz")
endif()

if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/opencv-src/lib")
  FetchContent_Declare(
    opencv
    URL ${OPENCV_URL}
  )
  FetchContent_MakeAvailable(opencv)
  message("Content downloaded from ${OPENCV_URL} to ${opencv_SOURCE_DIR}")
endif()
set(OPENCV_ROOT ${BUILD_DOWNLOAD_DIR}/opencv-src)

set(OPENCV_INCLUDES
  ${OPENCV_ROOT}/include/
  ${OPENCV_ROOT}/include/opencv/
)

set(OPENCV_LIBS_IMCODEC ${OPENCV_ROOT}/lib/libopencv_core.so
                        ${OPENCV_ROOT}/lib/libopencv_imgproc.so
                        ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so)

set(OPENCV_LIBS_IMCODEC_STATIC ${OPENCV_ROOT}/lib/libopencv_imgproc.a
                               ${OPENCV_ROOT}/lib/libopencv_imgcodecs.a
                               ${OPENCV_ROOT}/lib/libopencv_core.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/liblibjpeg-turbo.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/liblibpng.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/liblibopenjp2.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/liblibtiff.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/liblibwebp.a
                               ${OPENCV_ROOT}/lib/opencv4/3rdparty/libzlib.a)

if(NOT "${ARCH}" STREQUAL "riscv" AND NOT "${ARCH}" STREQUAL "RISCV")
  set(OPENCV_LIBS_IMCODEC_STATIC ${OPENCV_LIBS_IMCODEC_STATIC} ${OPENCV_ROOT}/lib/opencv4/3rdparty/libtegra_hal.a)
  set(OPENCV_LIBS_IMCODEC_STATIC ${OPENCV_LIBS_IMCODEC_STATIC} ${OPENCV_ROOT}/lib/opencv4/3rdparty/libittnotify.a)
endif()

set(OPENCV_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/opencv)


file(GLOB OPENCV_LIBS_CORE "${OPENCV_ROOT}/lib/libopencv_core.so*")
install(FILES ${OPENCV_LIBS_CORE} DESTINATION ${OPENCV_PATH}/lib)
file(GLOB OPENCV_LIBS_PROC "${OPENCV_ROOT}/lib/libopencv_imgproc.so*")
install(FILES ${OPENCV_LIBS_PROC} DESTINATION ${OPENCV_PATH}/lib)
file(GLOB OPENCV_LIBS_CODECS "${OPENCV_ROOT}/lib/libopencv_imgcodecs.so*")
install(FILES ${OPENCV_LIBS_CODECS} DESTINATION ${OPENCV_PATH}/lib)

install(DIRECTORY ${OPENCV_ROOT}/include/ DESTINATION ${OPENCV_PATH}/include)
