# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


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

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${OPENCV_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/opencv)
  install(FILES ${OPENCV_LIBS_MIN} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()