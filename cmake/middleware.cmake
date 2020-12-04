# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${MIDDLEWARE_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MIDDLEWARE_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MIDDLEWARE_SDK_ROOT}")
  message("-- Found MIDDLEWARE_SDK_ROOT (directory: ${MIDDLEWARE_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MIDDLEWARE_SDK_ROOT} is not a valid folder.")
endif()

set(MIDDLEWARE_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include/
)

set(MIDDLEWARE_LIBS ${MIDDLEWARE_SDK_ROOT}/lib/libcvitracer.so
                    ${MIDDLEWARE_SDK_ROOT}/lib/libini.so
                    ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
                    ${MIDDLEWARE_SDK_ROOT}/lib/libvpu.so
                    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so)

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/middleware)
  install(FILES ${MIDDLEWARE_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()
