# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if("${PREBUILT_DIR}" STREQUAL "")
    message(FATAL_ERROR "PREBUILT_DIR is not set!")
endif()

# LIBDEP_DIRS
# If any of these is not set, it'll use the prebuilt version.
if(USE_BSPSDK)
    set(LIBDEP_MIDDLEWARE_DIR ${BSPSDK_ROOT_DIR})
    set(LIBDEP_BMTAP2_DIR ${BSPSDK_ROOT_DIR})
    set(LIBDEP_BMVID_DIR ${BSPSDK_ROOT_DIR})
    set(LIBDEP_FFMPEG_DIR ${BSPSDK_ROOT_DIR})
    set(LIBDEP_OPENCV_DIR ${BSPSDK_ROOT_DIR})
endif()
# LIBDEP_GLOG_DIR
# LIBDEP_GTEST_DIR
# LIBDEP_PROTOBUF_DIR
# LIBDEP_OPENBLAS_DIR

# Prebuilt library lists
include(cmake/libdep/middleware-soc.cmake)
include(cmake/libdep/bmtap2.cmake)
include(cmake/libdep/glog.cmake)
include(cmake/libdep/gtest.cmake)
include(cmake/libdep/protobuf.cmake)
include(cmake/libdep/openblas.cmake)