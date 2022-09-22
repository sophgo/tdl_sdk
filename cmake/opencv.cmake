# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

set(FTP_SERVER_IP $ENV{FTP_SERVER_IP})
if (NOT FTP_SERVER_IP)
  set(FTP_SERVER_IP "10.80.0.5/sw_rls")
endif()

if (SHRINK_OPENCV_SIZE)

  if ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-uclibc-linux.cmake")
    set(OPENCV_URL ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/latest/uclibc/opencv_aisdk.tar.gz)
  elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-gnueabihf-linux.cmake")
    set(OPENCV_URL ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/latest/32bit/opencv_aisdk.tar.gz)
  elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-aarch64-linux.cmake")
    set(OPENCV_URL ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/latest/64bit/opencv_aisdk.tar.gz)
  elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-riscv64-linux.cmake")
    set(OPENCV_URL ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/latest/glibc_riscv64/opencv_aisdk.tar.gz)
  elseif("${CMAKE_TOOLCHAIN_FILE}" MATCHES "toolchain-riscv64-musl.cmake")
    set(OPENCV_URL ftp://swftp:cvitek@${FTP_SERVER_IP}/third_party/latest/musl_riscv64/opencv_aisdk.tar.gz)
  else()
    message(FATAL_ERROR "No shrinked opencv library for ${CMAKE_TOOLCHAIN_FILE}")
  endif()

  include(FetchContent)

  FetchContent_Declare(
    opencv
    URL ${OPENCV_URL}
  )
  FetchContent_MakeAvailable(opencv)
  message("Content downloaded from ${OPENCV_URL} to ${opencv_SOURCE_DIR}")

  set(OPENCV_ROOT ${opencv_SOURCE_DIR})

  set(OPENCV_INCLUDES
    ${OPENCV_ROOT}/include/
    ${OPENCV_ROOT}/include/opencv/
  )

  if ("${CVI_SYSTEM_PROCESSOR}" STREQUAL "RISCV")
    set(OPENCV_LIBS_MIN ${OPENCV_ROOT}/lib/libopencv_core.a)
    set(OPENCV_LIBS_IMGPROC ${OPENCV_ROOT}/lib/libopencv_imgproc.a)
    set(OPENCV_LIBS_IMCODEC ${OPENCV_ROOT}/lib/libopencv_core.so ${OPENCV_ROOT}/lib/libopencv_imgproc.so ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so)
  else()
    set(OPENCV_LIBS_MIN ${OPENCV_ROOT}/lib/libopencv_core.a
                      ${OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libtegra_hal.a)
    set(OPENCV_LIBS_IMGPROC ${OPENCV_ROOT}/lib/libopencv_imgproc.a)
    set(OPENCV_LIBS_IMCODEC ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    install(DIRECTORY ${OPENCV_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/opencv)
    install(FILES ${OPENCV_LIBS_MIN} ${OPENCV_LIBS_IMGPROC} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
  endif()
else()
  if("${OPENCV_ROOT}" STREQUAL "")
    message(FATAL_ERROR "You must set OPENCV_ROOT before building library.")
  elseif(EXISTS "${OPENCV_ROOT}")
    message("-- Found OPENCV_ROOT (directory: ${OPENCV_ROOT})")
  else()
    message(FATAL_ERROR "${OPENCV_ROOT} is not a valid folder.")
  endif()

  set(OPENCV_INCLUDES
      ${OPENCV_ROOT}/include/
      ${OPENCV_ROOT}/include/opencv/
  )

  set(OPENCV_LIBS_MIN ${OPENCV_ROOT}/lib/libopencv_core.so
                      ${OPENCV_ROOT}/lib/libopencv_imgproc.so)

  set(OPENCV_LIBS_IMCODEC ${OPENCV_ROOT}/lib/libopencv_imgcodecs.so)
endif()
