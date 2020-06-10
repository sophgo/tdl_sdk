# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if("${LIBDEP_MIDDLEWARE_DIR}" STREQUAL "")
    set(LIBDEP_MIDDLEWARE_DIR "${PREBUILT_DIR}/middleware-soc")
endif()
include(${CMAKE_SOURCE_DIR}/cmake/libdep/middleware-soc/bmvid.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/libdep/middleware-soc/ffmpeg.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/libdep/middleware-soc/opencv.cmake)  # soc opencv depends on above libraries!