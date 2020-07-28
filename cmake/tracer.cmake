# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if("${TRACER_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set TRACER_SDK_ROOT before building CVIAI library.")
elseif(EXISTS "${TRACER_SDK_ROOT}")
  message("-- Found TRACER_SDK_ROOT (directory: ${TRACER_SDK_ROOT})")
else()
  message(FATAL_ERROR "${TRACER_SDK_ROOT} is not a valid folder.")
endif()

project(tracer-sdk)
set(TRACER_INCLUDES
    ${TRACER_SDK_ROOT}/include/tracer
)

set(TRACER_LIBS
    ${TRACER_SDK_ROOT}/lib/libcvitracer.so
)

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${TRACER_SDK_ROOT}/include/tracer DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tracer)
  install(FILES ${TRACER_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()