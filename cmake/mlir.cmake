if("${MLIR_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MLIR_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MLIR_SDK_ROOT}")
  message("-- Found MLIR_SDK_ROOT (directory: ${MLIR_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MLIR_SDK_ROOT} is not a valid folder.")
endif()

if("${CVI_PLATFORM}" STREQUAL "CV186X")
  set(MLIR_INCLUDES ${SYSTEM_OUT_DIR}/usr/lib/libsophon-0.4.9/include/)
else()
  set(MLIR_INCLUDES ${MLIR_SDK_ROOT}/include/)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV186X")
  set(MLIR_LIBS
      ${SYSTEM_OUT_DIR}/usr/lib/libsophon-0.4.9/lib/libbmrt.so
      ${SYSTEM_OUT_DIR}/usr/lib/libsophon-0.4.9/lib/libbmlib.so
  )
else()
  set(MLIR_LIBS
      ${MLIR_SDK_ROOT}/lib/libcnpy.so
      ${MLIR_SDK_ROOT}/lib/libcvikernel.so
      ${MLIR_SDK_ROOT}/lib/libcvimath.so
      ${MLIR_SDK_ROOT}/lib/libcviruntime.so
  )
endif()

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${MLIR_SDK_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/sample/3rd/include/tpu)
  install(FILES ${MLIR_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/sample/3rd/lib)
endif()