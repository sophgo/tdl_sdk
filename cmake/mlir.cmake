# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if("${MLIR_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MLIR_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MLIR_SDK_ROOT}")
  message("-- Found MLIR_SDK_ROOT (directory: ${MLIR_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MLIR_SDK_ROOT} is not a valid folder.")
endif()

project(mlir-sdk)
set(MLIR_INCLUDES
    ${MLIR_SDK_ROOT}/include/
)

if("${CVI_TARGET}" STREQUAL "soc")
    set(CVI_CMODEL)
else()
    set(CVI_CMODEL ${MLIR_SDK_ROOT}/lib/libcvicmodel.so)
endif()

set(MLIR_LIBS
    ${MLIR_SDK_ROOT}/lib/libcvikernel.so
    ${MLIR_SDK_ROOT}/lib/libcvimath.so
    ${MLIR_SDK_ROOT}/lib/libcviruntime.so
    ${CVI_CMODEL}
)

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${MLIR_SDK_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
  install(FILES ${MLIR_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()